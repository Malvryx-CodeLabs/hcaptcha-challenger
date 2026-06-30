# -*- coding: utf-8 -*-
# Time       : 2022/2/15 17:43
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import os
from pathlib import Path

from hcaptcha_challenger import models as types
from hcaptcha_challenger.agent.challenger import AgentV, AgentConfig
from hcaptcha_challenger.agent.collector import Collector, CollectorConfig
from hcaptcha_challenger.models import (
    RequestType,
    CaptchaResponse,
    ChallengeTypeEnum,
    FastShotModelType,
    SCoTModelType,
    GroqModelType,
    LLMProvider,
    CoordinateGrid,
)
from hcaptcha_challenger.tools import ChallengeClassifier
from hcaptcha_challenger.tools import ImageClassifier
from hcaptcha_challenger.tools import SpatialBboxReasoner
from hcaptcha_challenger.tools import SpatialPathReasoner
from hcaptcha_challenger.tools import SpatialPointReasoner
from hcaptcha_challenger.utils import init_log

__all__ = [
    "ChallengeTypeEnum",
    "FastShotModelType",
    "SCoTModelType",
    "GroqModelType",
    "LLMProvider",
    "CoordinateGrid",
    "RequestType",
    "AgentV",
    "AgentConfig",
    "ImageClassifier",
    'ChallengeClassifier',
    'SpatialPathReasoner',
    'SpatialPointReasoner',
    'SpatialBboxReasoner',
    "CaptchaResponse",
    "Collector",
    "CollectorConfig",
    "types",
]

# Log location. Defaults to a folder next to the package, but that path is
# read-only when the package is installed system-wide and the process runs as a
# non-root user (e.g. in a container). Set HCAPTCHA_LOG_DIR to a writable path
# to override.
_LOG_BASE = os.environ.get("HCAPTCHA_LOG_DIR")
if _LOG_BASE:
    LOG_DIR = Path(_LOG_BASE).joinpath("{time:YYYY-MM-DD}")
else:
    LOG_DIR = Path(__file__).parent.joinpath("logs", "{time:YYYY-MM-DD}")

init_log(
    runtime=LOG_DIR.joinpath("runtime.log"),
    error=LOG_DIR.joinpath("error.log"),
    serialize=LOG_DIR.joinpath("serialize.log"),
)
