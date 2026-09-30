"""Context management: token estimation, message windowing, and compaction.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

from openagent.context.compactor import Compactor
from openagent.context.estimator import TokenEstimator
from openagent.context.messages import MessageManager

__all__ = [
    "Compactor",
    "MessageManager",
    "TokenEstimator",
]
