"""Hierarchical configuration loader and data models for OpenAgent.

Supports TOML configuration files, environment variable overrides, and CLI flags.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

from openagent.tools.mcp.client import MCPServerConfig
from openagent.tools.registry import DEFAULT_DANGER_POLICIES, PermissionAction


def _normalize_danger_policy(val: Mapping[str, Any] | None) -> dict[str, PermissionAction]:
    """Normalize a danger policy dictionary to PermissionAction enums."""
    policy = dict(DEFAULT_DANGER_POLICIES)
    if not val:
        return policy
    for k, v in val.items():
        if isinstance(v, PermissionAction):
            policy[str(k)] = v
        elif isinstance(v, str):
            with contextlib.suppress(ValueError):
                policy[str(k)] = PermissionAction(v.lower().strip())
    return policy


def _parse_mcp_servers(val: Any) -> list[MCPServerConfig]:
    """Parse MCP servers from list of tables or dict of tables."""
    if not val:
        return []
    servers: list[MCPServerConfig] = []
    if isinstance(val, list):
        for item in val:
            if isinstance(item, MCPServerConfig):
                servers.append(item)
            elif isinstance(item, Mapping):
                servers.append(MCPServerConfig.from_dict(item))
    elif isinstance(val, Mapping):
        for name, item in val.items():
            if isinstance(item, MCPServerConfig):
                servers.append(item)
            elif isinstance(item, Mapping):
                servers.append(MCPServerConfig.from_dict(item, name=str(name)))
    return servers


@dataclass
class OpenAgentConfig:
    """User and system runtime configuration for OpenAgent."""

    model: str = "gpt-4o"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_tool_iterations: int = 25
    danger_policy: dict[str, PermissionAction] = field(
        default_factory=lambda: dict(DEFAULT_DANGER_POLICIES)
    )
    mcp_servers: list[MCPServerConfig] = field(default_factory=list)
    workspace: Path = field(default_factory=lambda: Path.cwd().resolve())
    extra_instructions: list[str] = field(default_factory=list)
    auto_approve: bool = False
    session_id: str | None = None
    config_path: Path | None = None

    def __post_init__(self) -> None:
        if isinstance(self.workspace, str):
            self.workspace = Path(self.workspace).resolve()
        else:
            self.workspace = self.workspace.resolve()

        if isinstance(self.danger_policy, Mapping):
            self.danger_policy = _normalize_danger_policy(self.danger_policy)


def find_global_config_path() -> Path | None:
    """Find existing global configuration path, if any."""
    env_global = os.environ.get("OPENAGENT_GLOBAL_CONFIG")
    if env_global:
        p = Path(env_global).expanduser().resolve()
        return p

    candidates = [
        Path.home() / ".config" / "openagent" / "config.toml",
        Path.home() / ".openagent" / "config.toml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def find_project_config_path(start_dir: str | Path | None = None) -> Path | None:
    """Find local project configuration path searching upwards from start_dir."""
    curr = Path(start_dir or Path.cwd()).resolve()
    candidates = ["openagent.toml", ".openagent.toml"]
    for path in [curr, *curr.parents]:
        for candidate in candidates:
            target = path / candidate
            if target.is_file():
                return target.resolve()
    return None


def get_config_locations() -> list[tuple[str, Path, bool]]:
    """Get list of standard config file paths and whether they exist."""
    locations: list[tuple[str, Path, bool]] = []

    # Global
    global_path = find_global_config_path() or (Path.home() / ".openagent" / "config.toml")
    locations.append(("Global", global_path, global_path.is_file()))

    # Project
    proj_path = find_project_config_path() or (Path.cwd() / "openagent.toml")
    locations.append(("Project", proj_path, proj_path.is_file()))

    return locations


def _parse_toml_file(path: Path) -> dict[str, Any]:
    """Safely parse a TOML file returning a dict."""
    if not path.is_file():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def _apply_dict_to_config(config_dict: dict[str, Any], current: dict[str, Any]) -> None:
    """Merge dictionary parsed from TOML into current configuration dictionary."""
    scalar_keys = [
        "model",
        "api_key",
        "base_url",
        "temperature",
        "top_p",
        "max_tokens",
        "max_tool_iterations",
        "auto_approve",
        "session_id",
    ]
    for key in scalar_keys:
        if key in config_dict and config_dict[key] is not None:
            current[key] = config_dict[key]

    if "workspace" in config_dict and config_dict["workspace"] is not None:
        current["workspace"] = Path(config_dict["workspace"]).resolve()

    if "danger_policy" in config_dict and isinstance(config_dict["danger_policy"], Mapping):
        current_policy = dict(current.get("danger_policy", {}))
        current_policy.update(_normalize_danger_policy(config_dict["danger_policy"]))
        current["danger_policy"] = current_policy

    if "mcp_servers" in config_dict:
        current["mcp_servers"] = _parse_mcp_servers(config_dict["mcp_servers"])

    if "extra_instructions" in config_dict:
        val = config_dict["extra_instructions"]
        if isinstance(val, str):
            current["extra_instructions"] = [val]
        elif isinstance(val, list):
            current["extra_instructions"] = [str(x) for x in val]


def load_config(
    config_path: str | Path | None = None,
    **overrides: Any,
) -> OpenAgentConfig:
    """Load configuration hierarchically from files, environment variables, and overrides.

    Precedence order (highest to lowest):
    1. Explicit keyword arguments / CLI overrides
    2. Environment variables
    3. Custom config file (if specified via config_path)
    4. Project local config file (`openagent.toml` or `.openagent.toml`)
    5. Global user config file (`~/.config/openagent/config.toml` or `~/.openagent/config.toml`)
    6. Built-in defaults
    """
    config_data: dict[str, Any] = {
        "model": "gpt-4o",
        "api_key": None,
        "base_url": None,
        "temperature": None,
        "top_p": None,
        "max_tokens": None,
        "max_tool_iterations": 25,
        "danger_policy": dict(DEFAULT_DANGER_POLICIES),
        "mcp_servers": [],
        "workspace": Path.cwd().resolve(),
        "extra_instructions": [],
        "auto_approve": False,
        "session_id": None,
        "config_path": None,
    }

    # 1. Global config file
    global_p = find_global_config_path()
    if global_p and global_p.is_file():
        global_dict = _parse_toml_file(global_p)
        _apply_dict_to_config(global_dict, config_data)
        config_data["config_path"] = global_p

    # 2. Project local config file
    proj_p = find_project_config_path()
    if proj_p and proj_p.is_file():
        proj_dict = _parse_toml_file(proj_p)
        _apply_dict_to_config(proj_dict, config_data)
        config_data["config_path"] = proj_p

    # 3. Explicit config file if specified
    if config_path is not None:
        custom_p = Path(config_path).expanduser().resolve()
        if custom_p.is_file():
            custom_dict = _parse_toml_file(custom_p)
            _apply_dict_to_config(custom_dict, config_data)
            config_data["config_path"] = custom_p
        else:
            raise FileNotFoundError(f"Configuration file not found: {custom_p}")

    # 4. Environment variables
    if os.environ.get("OPENAGENT_MODEL"):
        config_data["model"] = os.environ["OPENAGENT_MODEL"]
    if os.environ.get("OPENAGENT_BASE_URL"):
        config_data["base_url"] = os.environ["OPENAGENT_BASE_URL"]
    if os.environ.get("OPENAGENT_API_KEY"):
        config_data["api_key"] = os.environ["OPENAGENT_API_KEY"]
    if os.environ.get("OPENAGENT_TEMPERATURE"):
        with contextlib.suppress(ValueError):
            config_data["temperature"] = float(os.environ["OPENAGENT_TEMPERATURE"])
    if os.environ.get("OPENAGENT_TOP_P"):
        with contextlib.suppress(ValueError):
            config_data["top_p"] = float(os.environ["OPENAGENT_TOP_P"])
    if os.environ.get("OPENAGENT_MAX_TOKENS"):
        with contextlib.suppress(ValueError):
            config_data["max_tokens"] = int(os.environ["OPENAGENT_MAX_TOKENS"])
    if os.environ.get("OPENAGENT_MAX_TOOL_ITERATIONS"):
        with contextlib.suppress(ValueError):
            config_data["max_tool_iterations"] = int(os.environ["OPENAGENT_MAX_TOOL_ITERATIONS"])
    if os.environ.get("OPENAGENT_WORKSPACE"):
        config_data["workspace"] = Path(os.environ["OPENAGENT_WORKSPACE"]).resolve()
    if os.environ.get("OPENAGENT_YES") or os.environ.get("OPENAGENT_AUTO_APPROVE"):
        val = (os.environ.get("OPENAGENT_YES") or os.environ.get("OPENAGENT_AUTO_APPROVE", "")).lower()
        config_data["auto_approve"] = val in ("1", "true", "yes", "on")

    # 5. Explicit overrides
    for k, v in overrides.items():
        if v is not None:
            if k == "workspace":
                config_data[k] = Path(v).resolve()
            elif k == "danger_policy" and isinstance(v, Mapping):
                current_p = dict(config_data.get("danger_policy", {}))
                current_p.update(_normalize_danger_policy(v))
                config_data[k] = current_p
            else:
                config_data[k] = v

    return OpenAgentConfig(
        model=config_data["model"],
        api_key=config_data["api_key"],
        base_url=config_data["base_url"],
        temperature=config_data["temperature"],
        top_p=config_data["top_p"],
        max_tokens=config_data["max_tokens"],
        max_tool_iterations=config_data["max_tool_iterations"],
        danger_policy=config_data["danger_policy"],
        mcp_servers=config_data["mcp_servers"],
        workspace=config_data["workspace"],
        extra_instructions=config_data["extra_instructions"],
        auto_approve=config_data["auto_approve"],
        session_id=config_data["session_id"],
        config_path=config_data["config_path"],
    )


def save_config(config: OpenAgentConfig, target_path: str | Path) -> None:
    """Save an OpenAgentConfig instance to a TOML file."""
    path = Path(target_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# OpenAgent configuration file",
        f"model = {json.dumps(config.model)}",
    ]

    if config.base_url:
        lines.append(f"base_url = {json.dumps(config.base_url)}")
    if config.api_key:
        lines.append(f"api_key = {json.dumps(config.api_key)}")
    if config.temperature is not None:
        lines.append(f"temperature = {config.temperature}")
    if config.top_p is not None:
        lines.append(f"top_p = {config.top_p}")
    if config.max_tokens is not None:
        lines.append(f"max_tokens = {config.max_tokens}")
    lines.append(f"max_tool_iterations = {config.max_tool_iterations}")
    lines.append(f"auto_approve = {'true' if config.auto_approve else 'false'}")
    lines.append(f"workspace = {json.dumps(config.workspace.as_posix())}")

    if config.extra_instructions:
        escaped_instr = [json.dumps(x) for x in config.extra_instructions]
        lines.append(f"extra_instructions = [{', '.join(escaped_instr)}]")

    # Danger policy table
    lines.append("\n[danger_policy]")
    for danger, action in config.danger_policy.items():
        act_val = action.value if isinstance(action, PermissionAction) else str(action)
        lines.append(f"{danger} = {json.dumps(act_val)}")

    # MCP servers
    for server in config.mcp_servers:
        lines.append("\n[[mcp_servers]]")
        lines.append(f"name = {json.dumps(server.name)}")
        lines.append(f"transport = {json.dumps(server.transport)}")
        if server.command:
            lines.append(f"command = {json.dumps(server.command)}")
        if server.args:
            args_str = ", ".join(json.dumps(a) for a in server.args)
            lines.append(f"args = [{args_str}]")
        if server.url:
            lines.append(f"url = {json.dumps(server.url)}")
        if server.cwd:
            lines.append(f"cwd = {json.dumps(Path(server.cwd).as_posix())}")
        if server.env:
            items = ", ".join(f"{json.dumps(k)} = {json.dumps(str(v))}" for k, v in server.env.items())
            lines.append(f"env = {{ {items} }}")

    content = "\n".join(lines) + "\n"
    path.write_text(content, encoding="utf-8")
