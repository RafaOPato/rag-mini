"""Armazenamento de vetores: `memory` (Python puro) e `pgvector` (Postgres)."""
import math
import os
import re
from dataclasses import dataclass

# 127.0.0.1 e não "localhost": no Windows, "localhost" tenta ::1 primeiro e o Docker só publica em IPv4.
DEFAULT_DSN = "postgresql://rag:rag@127.0.0.1:5433/rag"


@dataclass
class Hit:
    score: float  # similaridade de cosseno (1 = idêntico)
    source: str
    section: str
    index: int
    text: str


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class MemoryStore:
    """Tudo em listas Python. Não persiste: vive só enquanto o processo vive."""

    def __init__(self):
        self.rows = []

    def reset(self, dim: int):
        self.rows = []

    def add(self, chunks, vectors):
        self.rows += [(c, v) for c, v in zip(chunks, vectors)]

    def search(self, vector: list[float], k: int = 3) -> list[Hit]:
        scored = sorted(((cosine(vector, v), c) for c, v in self.rows), key=lambda t: -t[0])
        return [Hit(s, c.source, c.section, c.index, c.text) for s, c in scored[:k]]


class PgVectorStore:
    """Postgres + extensão pgvector. Distância de cosseno (`<=>`) com índice HNSW."""

    def __init__(self, dsn: str | None = None, table: str = "rag_mini_chunks"):
        try:
            import psycopg
        except ImportError:
            raise SystemExit("psycopg não instalado: pip install -r requirements.txt (ou use --store memory)")

        if not re.fullmatch(r"[a-z_][a-z0-9_]*", table):
            raise ValueError("nome de tabela inválido")
        self.table = table
        self._psycopg = psycopg
        try:
            self.conn = psycopg.connect(dsn or os.environ.get("RAG_DATABASE_URL") or DEFAULT_DSN, autocommit=True, connect_timeout=10)
        except psycopg.OperationalError as e:
            raise SystemExit(
                f"Não consegui conectar no Postgres ({e}).\n"
                "Suba o banco com: docker compose up -d --wait   (ou use --store memory)"
            )

    def reset(self, dim: int):
        t = self.table
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.conn.execute(f"DROP TABLE IF EXISTS {t}")
        self.conn.execute(
            f"CREATE TABLE {t} (id serial PRIMARY KEY, source text, section text, "
            f"chunk_index int, text text, embedding vector({int(dim)}))"
        )
        self.conn.execute(f"CREATE INDEX ON {t} USING hnsw (embedding vector_cosine_ops)")

    def add(self, chunks, vectors):
        with self.conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {self.table} (source, section, chunk_index, text, embedding) "
                "VALUES (%s, %s, %s, %s, %s::vector)",
                [(c.source, c.section, c.index, c.text, _literal(v)) for c, v in zip(chunks, vectors)],
            )

    def search(self, vector: list[float], k: int = 3) -> list[Hit]:
        lit = _literal(vector)
        try:
            rows = self.conn.execute(
                f"SELECT 1 - (embedding <=> %s::vector), source, section, chunk_index, text "
                f"FROM {self.table} ORDER BY embedding <=> %s::vector LIMIT %s",
                (lit, lit, k),
            ).fetchall()
        except self._psycopg.errors.UndefinedTable:
            raise SystemExit("A tabela não existe ainda. Rode primeiro: python ingest.py")
        except self._psycopg.errors.DataException as e:  # dimensões diferentes
            raise SystemExit(f"{e}\nO índice foi feito com outro embedder. Rode de novo: python ingest.py")
        # Vetor zero (pergunta só de stopwords) dá NaN no `<=>`; o MemoryStore devolve 0.0 nesse caso.
        return [Hit(0.0 if math.isnan(s) else float(s), src, sec, i, t) for s, src, sec, i, t in rows]


def _literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{x:.7f}" for x in vector) + "]"
