"""YAML configuration loader for agno-docs-agent.

Loads settings from ``config.yml`` with env var overrides.
Allows switching model providers, embedders, and chunking strategies
without code changes.

Supports: ollama (local), openrouter, openai, anthropic, google.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Load .env if present
_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
if _ENV_PATH.exists():
    with open(_ENV_PATH, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key not in os.environ:
                    os.environ[key] = value


@dataclass
class ModelConfig:
    provider: str = "ollama"
    name: str = "gemma4:26b"
    temperature: float = 0.3
    max_tokens: int = 2048

    def to_model_string(self) -> str:
        """Return Agno-compatible model string (provider:model_id)."""
        return f"{self.provider}:{self.name}"


@dataclass
class EmbedderConfig:
    provider: str = "sentence_transformer"
    model: str = "all-MiniLM-L6-v2"
    dimensions: int = 384


@dataclass
class ChunkingConfig:
    strategy: str = "markdown"
    chunk_size: int = 512
    overlap: int = 100


@dataclass
class KnowledgeConfig:
    db_path: str = "knowledge/agno_agent.db"
    llms_path: str = "knowledge/llms-full.txt"
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)


@dataclass
class ServerConfig:
    transport: str = "stdio"
    docs_path: str = ""


@dataclass
class LoggingConfig:
    level: str = "INFO"


@dataclass
class AgentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path = "config.yml") -> AgentConfig:
        """Load config from YAML file, with env var overrides."""
        cfg_path = Path(path)
        if not cfg_path.exists():
            return cls()

        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

        # Env var overrides
        if os.environ.get("AGNO_MODEL"):
            raw.setdefault("model", {})["name"] = os.environ["AGNO_MODEL"]
        if os.environ.get("AGNO_DOCS_PATH"):
            raw.setdefault("server", {})["docs_path"] = os.environ["AGNO_DOCS_PATH"]
        if os.environ.get("AGNO_EMBED_MODEL"):
            raw.setdefault("embedder", {})["model"] = os.environ["AGNO_EMBED_MODEL"]

        return cls(
            model=ModelConfig(**raw.get("model", {})),
            embedder=EmbedderConfig(**raw.get("embedder", {})),
            knowledge=KnowledgeConfig(
                **{
                    **raw.get("knowledge", {}),
                    "chunking": ChunkingConfig(**raw.get("knowledge", {}).get("chunking", {}))
                } if "chunking" in raw.get("knowledge", {})
                else {}
            ),
            server=ServerConfig(**raw.get("server", {})),
            logging=LoggingConfig(**raw.get("logging", {})),
        )

    @classmethod
    def from_env(cls) -> AgentConfig:
        """Quick config from environment variables only."""
        return cls(
            model=ModelConfig(
                name=os.environ.get("AGNO_MODEL", "ollama:gemma4:26b"),
            ),
            server=ServerConfig(
                docs_path=os.environ.get("AGNO_DOCS_PATH", ""),
            ),
            embedder=EmbedderConfig(
                model=os.environ.get("AGNO_EMBED_MODEL", "all-MiniLM-L6-v2"),
            ),
        )


def _parse_recursive(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    """Recursively parse nested dicts into dataclass fields."""
    import dataclasses
    result: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        key = f.name
        if key in data:
            raw_val = data[key]
            if dataclasses.is_dataclass(f.type) and isinstance(raw_val, dict):
                result[key] = f.type(**_parse_recursive(f.type, raw_val))
            else:
                result[key] = raw_val
    return result


# Global config — loaded once at import time
config = AgentConfig.from_yaml()
