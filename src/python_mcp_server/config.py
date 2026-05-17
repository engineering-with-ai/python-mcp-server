"""Per-stage configuration loader.

One axis: ENV (local | beta) — picks the cfg.yml stage block. Each stage
holds a single provider settings block (BedrockSettings | OllamaSettings)
dispatched by the `llm_provider` discriminator. Pydantic validates the
right field combo at load time.

Connection URLs (GRAPH_URL / NEPTUNE_HOST / VECTOR_URL) are NOT held
here — they come from process env at consume time (compose env_file →
secrets.env populated by CFN-managed Secrets Manager).
"""

import enum
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, Field


class LogLevel(enum.StrEnum):
    """Stdlib logging level names; StrEnum so cfg.yml strings parse directly."""

    ERROR = "ERROR"
    WARN = "WARN"
    INFO = "INFO"
    DEBUG = "DEBUG"


class BedrockSettings(BaseModel):
    """Bedrock provider — cloud stages."""

    llm_provider: Literal["bedrock"]
    bedrock_chat_model_id: str
    bedrock_embedding_model_id: str
    vector_seed_url: str | None = None
    graph_neo4j_seed_url: str | None = None


class OllamaSettings(BaseModel):
    """Ollama provider — airgapped + local dev dogfood.

    Embedding model is Qwen3 (2560d) truncated to 1024 (Matryoshka) so
    schema matches Bedrock Titan dumps.
    """

    llm_provider: Literal["ollama"]
    ollama_base_url: str
    ollama_chat_model: str
    ollama_embedding_model: str
    vector_seed_url: str | None = None
    graph_neo4j_seed_url: str | None = None


ProviderSettings = Annotated[
    BedrockSettings | OllamaSettings, Field(discriminator="llm_provider")
]


class StageConfig(BaseModel):
    """One stage block in cfg.yml."""

    log_level: LogLevel
    e2e: bool = False  # tests-only: false → pook mocks HTTP; true → hit real services
    settings: ProviderSettings


class _ConfigMap(BaseModel):
    local: StageConfig
    beta: StageConfig


class Config(BaseModel):
    """Resolved runtime config for a single process."""

    log_level: LogLevel
    e2e: bool = False
    settings: ProviderSettings


def load_config() -> Config:
    """Resolve cfg[ENV] -> Config. ENV defaults to local."""
    cfg_path = Path(__file__).parent / "cfg.yml"
    with open(cfg_path) as file:
        config_map = _ConfigMap(**yaml.safe_load(file))
    env = os.environ.get("ENV", "local")
    stage = config_map.local if env != "beta" else config_map.beta
    return Config(log_level=stage.log_level, e2e=stage.e2e, settings=stage.settings)


class _ZuluFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return (
            f"{datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%S.%fZ')}  "
            f"{record.levelname}: {record.getMessage()}"
        )


def setup_logger(config: Config) -> None:
    """Color level names + Zulu timestamp formatter."""
    for ind, lvl in enumerate(
        [logging.ERROR, logging.INFO, logging.WARNING, logging.DEBUG]
    ):
        logging.addLevelName(
            lvl,
            f"\033[0;3{ind + 1}m%s\033[1;0m" % logging.getLevelName(lvl),
        )
    logging.basicConfig(
        level=config.log_level.value, handlers=[logging.StreamHandler()]
    )
    for handler in logging.root.handlers:
        handler.setFormatter(_ZuluFormatter())
