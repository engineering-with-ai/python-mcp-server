import enum
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel
import yaml


class LogLevel(enum.StrEnum):
    """Logging levels for the application."""

    ERROR = "ERROR"
    WARN = "WARN"
    INFO = "INFO"
    DEBUG = "DEBUG"


class Neo4jConfig(BaseModel):
    """Neo4j database configuration settings."""

    uri: str
    user: str
    database: str


class PostgresConfig(BaseModel):
    """PostgreSQL database configuration settings."""

    embeddings_table: str


class Config(BaseModel):
    """Application configuration settings."""

    log_level: LogLevel
    neo4j: Neo4jConfig
    postgres: PostgresConfig


class _ConfigMap(BaseModel):
    local: Config
    beta: Config


def load_config_from_env() -> Config:
    """Load configuration from environment variables only.

    Returns:
        Config object created from environment variables with sensible defaults

    Example:
        >>> config = load_config_from_env()
        >>> config.neo4j.uri
        'bolt://localhost:7687'

    """
    return Config(
        log_level=LogLevel(os.getenv("LOG_LEVEL", "INFO")),
        neo4j=Neo4jConfig(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
        ),
        postgres=PostgresConfig(
            embeddings_table=os.getenv("POSTGRES_TABLE", "embeddings"),
        ),
    )


def load_config() -> Config:
    """Load configuration from cfg.yml file based on environment.

    Reads the configuration file and returns the appropriate config
    based on the ENV environment variable (defaults to 'local').
    If cfg.yml is not found, falls back to environment variables.

    Returns:
        Config object for the current environment

    Raises:
        yaml.YAMLError: If cfg.yml contains invalid YAML

    Example:
        >>> config = load_config()  # Uses ENV=local by default
        >>> config.log_level
        <LogLevel.DEBUG: 'DEBUG'>

    """
    cfg_path = Path(__file__).parent.parent.parent / "cfg.yml"
    if not cfg_path.exists():
        # Fall back to environment variables if cfg.yml doesn't exist
        return load_config_from_env()

    with open(cfg_path) as file:
        config_map = _ConfigMap(**yaml.safe_load(file))
        environment = os.environ.get("ENV", "local")
        match environment:
            case "beta":
                return config_map.beta
            case _:
                return config_map.local


class _ZuluFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        """Format log records with Zulu time timestamps.

        Args:
            record: The log record to format

        Returns:
            Formatted log string with Zulu timestamp, level, and message

        Example:
            >>> formatter = _ZuluFormatter()
            >>> record = logging.LogRecord(...)
            >>> formatter.format(record)
            '2024-01-01T12:00:00.123456Z  INFO: Sample message'

        """
        zulu_time = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        return f"{zulu_time}  {record.levelname}: {record.getMessage()}"


def setup_logger(config: Config) -> None:
    """Set up the application logger with colored output and Zulu time formatting.

    Configures logging with color-coded level names and custom Zulu timestamp
    formatting using the provided configuration.

    Args:
        config: Configuration object containing log level settings

    Example:
        >>> cfg = Config(log_level=LogLevel.INFO, ...)
        >>> setup_logger(cfg)
        >>> logging.info("Test message")  # Outputs with colors and Zulu time

    """
    # adds color
    for ind, lvl in enumerate(
        [logging.ERROR, logging.INFO, logging.WARNING, logging.DEBUG],
    ):
        logging.addLevelName(
            lvl,
            f"\033[0;3{ind + 1}m%s\033[1;0m" % logging.getLevelName(lvl),
        )

    # inits logger
    logging.basicConfig(
        encoding="utf-8",
        level=config.log_level.value,
        handlers=[logging.StreamHandler()],
    )

    for handler in logging.root.handlers:
        handler.setFormatter(_ZuluFormatter())
