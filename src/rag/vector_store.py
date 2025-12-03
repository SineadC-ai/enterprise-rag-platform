from typing import List, Dict, Any, Optional

import psycopg2
from psycopg2.extras import execute_values

from src.core.settings import get_settings


class PgVectorStore:
    """
    pgvector-backed store for chunk embeddings.

    Uses a single table: chunk_embeddings.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._dsn = settings.database_url

    def _get_connection(self):
        return psycopg2.connect(self._dsn)

    def init_schema(self) -> None:
        """
        Creates the chunk_embeddings table and indexes if they do not exist.
        """
        ddl = """
        CREATE EXTENSION IF NOT EXISTS vector;

        CREATE TABLE IF NOT EXISTS chunk_embeddings (
            id BIGSERIAL PRIMARY KEY,
            document_id TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding VECTOR(16)
        );

        CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_document
            ON chunk_embeddings (document_id);

        CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_embedding
            ON chunk_embeddings
            USING ivfflat (embedding vector_cosine_ops);
        """
        conn = self._get_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(ddl)
        finally:
            conn.close()

    def upsert_embeddings(self, chunks_with_embeddings: List[Dict[str, Any]]) -> None:
        """
        Inserts chunk embeddings into the database.
        Performs simple inserts without de-duplication.
        """
        if not chunks_with_embeddings:
            return

        rows = []
        for chunk in chunks_with_embeddings:
            rows.append(
                (
                    chunk["document_id"],
                    chunk["chunk_id"],
                    int(chunk["index"]),
                    chunk["text"],
                    chunk["embedding"],
                )
            )

        sql = """
        INSERT INTO chunk_embeddings (
            document_id,
            chunk_id,
            chunk_index,
            text,
            embedding
        )
        VALUES %s
        """

        conn = self._get_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    execute_values(cur, sql, rows)
        finally:
            conn.close()

    def search_similar_chunks(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        document_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Returns the most similar chunks ordered by vector similarity.
        """
        # Convert embedding list to pgvector-compatible text format: '[0.1,0.2,...]'
        vec_str = "[" + ",".join(f"{x:.6f}" for x in query_embedding) + "]"

        conn = self._get_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    if document_id is not None:
                        sql = """
                        SELECT
                            document_id,
                            chunk_id,
                            chunk_index,
                            text,
                            1 - (embedding <=> %s::vector) AS similarity
                        FROM chunk_embeddings
                        WHERE document_id = %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """
                        params = [vec_str, document_id, vec_str, top_k]
                    else:
                        sql = """
                        SELECT
                            document_id,
                            chunk_id,
                            chunk_index,
                            text,
                            1 - (embedding <=> %s::vector) AS similarity
                        FROM chunk_embeddings
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """
                        params = [vec_str, vec_str, top_k]

                    cur.execute(sql, params)
                    rows = cur.fetchall()
        finally:
            conn.close()

        results: List[Dict[str, Any]] = []
        for row in rows:
            results.append(
                {
                    "document_id": row[0],
                    "chunk_id": row[1],
                    "chunk_index": row[2],
                    "text": row[3],
                    "score": float(row[4]),
                }
            )
        return results
