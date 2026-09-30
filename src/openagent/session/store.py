"""JSONL-based session persistence and metadata storage.

Provides atomic file persistence, history serialization with full fidelity for
multi-part content and tool calls, session listing, and session resumption.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openagent.core.types import Message

logger = logging.getLogger(__name__)


@dataclass
class SessionMetadata:
    """Metadata describing a saved conversation session."""

    session_id: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    model: str = ""
    title: str = ""
    message_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to a serializable dictionary."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "model": self.model,
            "title": self.title,
            "message_count": self.message_count,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SessionMetadata:
        """Construct SessionMetadata from a dictionary."""
        return cls(
            session_id=str(data.get("session_id", "")),
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
            model=str(data.get("model", "")),
            title=str(data.get("title", "")),
            message_count=int(data.get("message_count", 0)),
        )


class SessionStore:
    """Manages persistence of conversation histories in JSONL format with atomic writes."""

    def __init__(self, storage_dir: str | Path | None = None) -> None:
        if storage_dir is None:
            self._storage_dir = Path.home() / ".openagent" / "sessions"
        else:
            self._storage_dir = Path(storage_dir).resolve()
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    @property
    def storage_dir(self) -> Path:
        """Return the base storage directory."""
        return self._storage_dir

    def create_session(self, model: str = "", title: str | None = None) -> str:
        """Create a new session record and return its unique identifier.

        Args:
            model: The primary model identifier associated with this session.
            title: Human-readable title for the session.

        Returns:
            The generated session_id string.
        """
        session_id = uuid.uuid4().hex
        now = time.time()
        effective_title = title if title is not None else f"Session {session_id[:8]}"
        meta = SessionMetadata(
            session_id=session_id,
            created_at=now,
            updated_at=now,
            model=model,
            title=effective_title,
            message_count=0,
        )
        self._atomic_write_session(session_id, meta, [])
        return session_id

    def save_messages(
        self,
        session_id: str,
        messages: Sequence[Message],
        metadata: SessionMetadata | None = None,
    ) -> None:
        """Atomically persist a sequence of messages for a session.

        Args:
            session_id: The session identifier.
            messages: List of Message objects to persist.
            metadata: Optional updated SessionMetadata. If omitted, existing metadata
                      is updated with new message count and updated timestamp.
        """
        now = time.time()
        if metadata is None:
            target = self._get_session_path(session_id)
            if target.exists():
                try:
                    existing_meta, _ = self.load_session(session_id)
                    metadata = SessionMetadata(
                        session_id=session_id,
                        created_at=existing_meta.created_at,
                        updated_at=now,
                        model=existing_meta.model,
                        title=existing_meta.title,
                        message_count=len(messages),
                    )
                except Exception:
                    metadata = SessionMetadata(
                        session_id=session_id,
                        created_at=now,
                        updated_at=now,
                        model="",
                        title=f"Session {session_id[:8]}",
                        message_count=len(messages),
                    )
            else:
                metadata = SessionMetadata(
                    session_id=session_id,
                    created_at=now,
                    updated_at=now,
                    model="",
                    title=f"Session {session_id[:8]}",
                    message_count=len(messages),
                )
        else:
            metadata.message_count = len(messages)
            metadata.updated_at = now

        self._atomic_write_session(session_id, metadata, messages)

    def append_message(self, session_id: str, message: Message) -> None:
        """Append a single message to an existing session history atomically.

        Args:
            session_id: The target session identifier.
            message: The Message to append.

        Raises:
            FileNotFoundError: If the session does not exist.
        """
        meta, messages = self.load_session(session_id)
        messages.append(message)
        self.save_messages(session_id, messages, metadata=meta)

    def load_session(self, session_id: str) -> tuple[SessionMetadata, list[Message]]:
        """Load session metadata and message history from disk.

        Args:
            session_id: The session identifier to load.

        Returns:
            Tuple of (SessionMetadata, list of Message).

        Raises:
            FileNotFoundError: If the session file does not exist.
        """
        session_path = self._get_session_path(session_id)
        if not session_path.is_file():
            raise FileNotFoundError(f"Session '{session_id}' not found at {session_path}")

        lines: list[str] = []
        with open(session_path, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    lines.append(stripped)

        if not lines:
            now = time.time()
            meta = SessionMetadata(
                session_id=session_id,
                created_at=now,
                updated_at=now,
                model="",
                title=f"Session {session_id[:8]}",
                message_count=0,
            )
            return meta, []

        first_data = json.loads(lines[0])
        messages: list[Message] = []

        if isinstance(first_data, Mapping) and (
            first_data.get("type") == "metadata" or "session_id" in first_data
        ):
            meta = SessionMetadata.from_dict(first_data)
            message_lines = lines[1:]
        else:
            stat = session_path.stat()
            meta = SessionMetadata(
                session_id=session_id,
                created_at=stat.st_ctime,
                updated_at=stat.st_mtime,
                model="",
                title=f"Session {session_id[:8]}",
                message_count=0,
            )
            message_lines = lines

        for line in message_lines:
            raw_obj = json.loads(line)
            if isinstance(raw_obj, Mapping):
                messages.append(self._deserialize_message(raw_obj))

        meta.message_count = len(messages)
        return meta, messages

    def list_sessions(self) -> list[SessionMetadata]:
        """List all stored sessions sorted by updated timestamp descending.

        Returns:
            List of SessionMetadata records.
        """
        sessions: list[SessionMetadata] = []
        for file_path in self._storage_dir.glob("*.jsonl"):
            try:
                with open(file_path, encoding="utf-8") as f:
                    first_line = f.readline().strip()
                if not first_line:
                    stat = file_path.stat()
                    sessions.append(
                        SessionMetadata(
                            session_id=file_path.stem,
                            created_at=stat.st_ctime,
                            updated_at=stat.st_mtime,
                            model="",
                            title=file_path.stem,
                            message_count=0,
                        )
                    )
                    continue

                first_data = json.loads(first_line)
                if isinstance(first_data, Mapping) and (
                    first_data.get("type") == "metadata" or "session_id" in first_data
                ):
                    sessions.append(SessionMetadata.from_dict(first_data))
                else:
                    # Legacy or raw JSONL
                    stat = file_path.stat()
                    # Count total non-empty lines for message_count
                    with open(file_path, encoding="utf-8") as f_full:
                        count = sum(1 for ln in f_full if ln.strip())
                    sessions.append(
                        SessionMetadata(
                            session_id=file_path.stem,
                            created_at=stat.st_ctime,
                            updated_at=stat.st_mtime,
                            model="",
                            title=file_path.stem,
                            message_count=count,
                        )
                    )
            except Exception as exc:
                logger.warning("Failed to read session header from %s: %s", file_path, exc)
                continue

        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions

    def delete_session(self, session_id: str) -> bool:
        """Delete session data and any associated temporary files.

        Args:
            session_id: The session identifier to delete.

        Returns:
            True if session file was found and removed, False otherwise.
        """
        session_path = self._get_session_path(session_id)
        existed = False
        if session_path.is_file():
            session_path.unlink()
            existed = True

        # Clean up any leftover temporary files for this session
        for tmp_file in self._storage_dir.glob(f"{session_id}.*.tmp"):
            with contextlib.suppress(OSError):
                tmp_file.unlink(missing_ok=True)

        return existed

    def _get_session_path(self, session_id: str) -> Path:
        return self._storage_dir / f"{session_id}.jsonl"

    def _atomic_write_session(
        self,
        session_id: str,
        metadata: SessionMetadata,
        messages: Sequence[Message],
    ) -> None:
        """Atomically write metadata and messages to disk using a temp file."""
        target_path = self._get_session_path(session_id)
        tmp_path = self._storage_dir / f"{session_id}.{uuid.uuid4().hex[:8]}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                meta_dict = metadata.to_dict()
                meta_dict["type"] = "metadata"
                f.write(json.dumps(meta_dict, ensure_ascii=False) + "\n")
                for msg in messages:
                    msg_dict = self._serialize_message(msg)
                    f.write(json.dumps(msg_dict, ensure_ascii=False) + "\n")
            os.replace(tmp_path, target_path)
        except Exception:
            if tmp_path.exists():
                with contextlib.suppress(OSError):
                    tmp_path.unlink()
            raise

    @staticmethod
    def _serialize_message(msg: Message) -> dict[str, Any]:
        """Serialize a Message with full fidelity including ToolCall raw_arguments."""
        data = msg.to_dict()
        data["type"] = "message"
        if msg.tool_calls:
            data["tool_calls"] = [
                {
                    "id": tc.id,
                    "name": tc.name,
                    "arguments": dict(tc.arguments),
                    "raw_arguments": tc.raw_arguments,
                }
                for tc in msg.tool_calls
            ]
        return data

    @staticmethod
    def _deserialize_message(data: Mapping[str, Any]) -> Message:
        """Deserialize a dict into a Message restoring ToolCall raw_arguments."""
        payload = dict(data)
        if payload.get("type") == "message" and "data" in payload and isinstance(payload["data"], Mapping):
            payload = dict(payload["data"])

        msg = Message.from_dict(payload)
        raw_tool_calls = payload.get("tool_calls")
        if raw_tool_calls and isinstance(raw_tool_calls, Sequence) and msg.tool_calls:
            for tc, tc_dict in zip(msg.tool_calls, raw_tool_calls, strict=False):
                if isinstance(tc_dict, Mapping) and "raw_arguments" in tc_dict:
                    tc.raw_arguments = str(tc_dict["raw_arguments"])
        return msg
