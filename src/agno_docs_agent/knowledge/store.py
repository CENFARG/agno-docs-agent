"""Vector store backed by sqlite-vec, implemented as an Agno VectorDb ABC subclass.

Subclasses :class:`agno.vectordb.base.VectorDb` so it can be used directly
as an Agno ``Knowledge.vector_db``. Embeds documents internally via the
configured embedder.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import sqlite_vec  # type: ignore[import-untyped]
from agno.knowledge.document.base import Document
from agno.vectordb.base import VectorDb


@dataclass(frozen=True)
class SqliteVecDbConfig:
    """Configuration for the sqlite-vec vector store.

    Attributes:
        db_path: Filesystem path to the SQLite database file.
        table_name: Name of the vec0 virtual table (default: ``"docs"``).
        dimensions: Dimensionality of embedding vectors (default: ``384``).
    """

    db_path: str
    table_name: str = "docs"
    dimensions: int = 384


class SqliteVecDb(VectorDb):
    """A sqlite-vec backed VectorDb for cosine-similarity search.

    Args:
        config: SqliteVecDbConfig with path, table name, and dimensions.
        embedder: An object with a ``get_embedding(text) -> List[float]``
            method (e.g. Agno's ``SentenceTransformerEmbedder``).
        **kwargs: Forwarded to :class:`VectorDb.__init__`.
    """

    def __init__(
        self,
        config: SqliteVecDbConfig,
        embedder: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._config = config
        self._embedder = embedder
        self._conn: sqlite3.Connection | None = None

    # -- lifecycle ----------------------------------------------------------

    def create(self) -> None:
        """Create the vec0 virtual table. Idempotent."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        conn = sqlite3.connect(self._config.db_path, check_same_thread=False)
        conn.enable_load_extension(True)
        conn.load_extension(sqlite_vec.loadable_path())
        conn.execute(
            f"""CREATE VIRTUAL TABLE IF NOT EXISTS {self._config.table_name}
            USING vec0(embedding float[{self._config.dimensions}],
                       content TEXT,
                       content_hash TEXT partition key,
                       name TEXT,
                       meta_data TEXT)"""
        )
        conn.commit()
        self._conn = conn

    async def async_create(self) -> None:
        self.create()

    def exists(self) -> bool:
        """Check whether the vec0 table exists."""
        if not os.path.isfile(self._config.db_path):
            return False
        try:
            conn = sqlite3.connect(self._config.db_path, check_same_thread=False)
            conn.enable_load_extension(True)
            conn.load_extension(sqlite_vec.loadable_path())
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (self._config.table_name,),
            ).fetchone()
            conn.close()
            return row is not None
        except Exception:
            return False

    async def async_exists(self) -> bool:
        return self.exists()

    def drop(self) -> None:
        """Drop the vec0 table."""
        self._get_conn().execute(f"DROP TABLE IF EXISTS {self._config.table_name}")
        self._get_conn().commit()

    async def async_drop(self) -> None:
        self.drop()

    def delete(self) -> bool:
        """Delete the underlying database file."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        try:
            os.remove(self._config.db_path)
        except FileNotFoundError:
            pass
        return not os.path.isfile(self._config.db_path)

    # -- data ---------------------------------------------------------------

    def insert(
        self,
        content_hash: str,
        documents: List[Document],
        filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Embed and insert documents into the vector store."""
        conn = self._get_conn()
        with conn:
            for doc in documents:
                embedding = self._embed(doc.content)
                meta_json = json.dumps(doc.meta_data) if doc.meta_data else "{}"
                conn.execute(
                    f"INSERT INTO {self._config.table_name} "
                    "(content, content_hash, name, meta_data, embedding) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        doc.content,
                        content_hash,
                        doc.name or "",
                        meta_json,
                        sqlite_vec.serialize_float32(embedding),
                    ),
                )

    async def async_insert(
        self,
        content_hash: str,
        documents: List[Document],
        filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        import asyncio

        await asyncio.to_thread(self.insert, content_hash, documents, filters)

    def search(
        self,
        query: str,
        limit: int = 5,
        filters: Optional[Any] = None,
    ) -> List[Document]:
        """Search by semantic similarity; returns List[Document]."""
        q_emb = self._embed(query)
        serialized = sqlite_vec.serialize_float32(q_emb)
        rows = self._get_conn().execute(
            f"""SELECT content, content_hash, name, meta_data, distance
            FROM {self._config.table_name}
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?""",
            (serialized, limit),
        ).fetchall()
        results: List[Document] = []
        for row in rows:
            meta: dict[str, Any] = {}
            try:
                meta = json.loads(row[3]) if row[3] else {}
            except (json.JSONDecodeError, TypeError):
                pass
            results.append(
                Document(
                    content=row[0],
                    id=row[1],
                    name=row[2] or None,
                    meta_data=meta,
                )
            )
        return results

    async def async_search(
        self,
        query: str,
        limit: int = 5,
        filters: Optional[Any] = None,
    ) -> List[Document]:
        import asyncio

        return await asyncio.to_thread(self.search, query, limit, filters)

    # -- content hash / name ------------------------------------------------

    def content_hash_exists(self, content_hash: str) -> bool:
        row = self._get_conn().execute(
            f"SELECT 1 FROM {self._config.table_name} "
            "WHERE content_hash = ? LIMIT 1",
            (content_hash,),
        ).fetchone()
        return row is not None

    def name_exists(self, name: str) -> bool:
        row = self._get_conn().execute(
            f"SELECT 1 FROM {self._config.table_name} WHERE name = ? LIMIT 1",
            (name,),
        ).fetchone()
        return row is not None

    def async_name_exists(self, name: str) -> bool:
        return self.name_exists(name)

    # -- remaining abstract stubs (MVP: raise NotImplementedError) ----------

    def id_exists(self, id: str) -> bool:
        raise NotImplementedError

    def upsert(self, content_hash: str, documents: List[Document], filters: Optional[Dict[str, Any]] = None) -> None:
        raise NotImplementedError

    async def async_upsert(self, content_hash: str, documents: List[Document], filters: Optional[Dict[str, Any]] = None) -> None:
        raise NotImplementedError

    def delete_by_id(self, id: str) -> bool:
        raise NotImplementedError

    def delete_by_name(self, name: str) -> bool:
        raise NotImplementedError

    def delete_by_metadata(self, metadata: Dict[str, Any]) -> bool:
        raise NotImplementedError

    def delete_by_content_id(self, content_id: str) -> bool:
        raise NotImplementedError

    def get_supported_search_types(self) -> List[str]:
        raise NotImplementedError

    # -- internal -----------------------------------------------------------

    def _embed(self, text: str) -> List[float]:
        if self._embedder is None:
            raise RuntimeError("SqliteVecDb has no embedder configured")
        if hasattr(self._embedder, "get_embedding"):
            result: Any = self._embedder.get_embedding(text)
            return result  # type: ignore[no-any-return]
        result = self._embedder.embed(text)
        return result  # type: ignore[no-any-return]

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("SqliteVecDb not initialised — call create() first")
        return self._conn
