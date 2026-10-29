"""Cola do pipeline: argumentos de linha de comando, ingestão e busca."""
import argparse
import sys
from pathlib import Path

from .chunking import chunk_corpus
from .embedders import get_embedder
from .stores import MemoryStore, PgVectorStore

ROOT = Path(__file__).resolve().parent.parent


def setup_stdout():
    # No Windows o console costuma ser cp1252; os textos estão em UTF-8.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--store", choices=["pgvector", "memory"], default="pgvector",
                        help="onde ficam os vetores (padrão: pgvector; memory não precisa de Docker)")
    parser.add_argument("--embedder", choices=["auto", "fastembed", "hash"], default="auto",
                        help="auto = fastembed se estiver instalado, senão hash")
    parser.add_argument("--corpus", default=str(ROOT / "corpus"))
    parser.add_argument("--size", type=int, default=400, help="tamanho máximo do trecho, em caracteres")
    parser.add_argument("--overlap", type=int, default=150, help="sobreposição entre trechos, em caracteres")


def ingest(store, embedder, corpus_dir, size=400, overlap=150) -> int:
    """(Re)cria o índice do zero. Rodar duas vezes dá o mesmo resultado."""
    chunks = chunk_corpus(corpus_dir, size, overlap)
    if not chunks:  # erro de digitação em --corpus não pode apagar o índice que já existe
        raise SystemExit(f"nenhum .md com texto em {corpus_dir}: o índice não foi alterado")
    store.reset(embedder.dim)
    store.add(chunks, embedder.embed([c.embed_text for c in chunks]))
    return len(chunks)


def open_index(args):
    """Devolve (store, embedder) prontos para buscar.

    pgvector: só conecta (o índice vem do ingest.py).
    memory: não persiste, então indexa o corpus agora (leva segundos).
    """
    embedder = get_embedder(args.embedder)
    if args.store == "memory":
        store = MemoryStore()
        ingest(store, embedder, args.corpus, args.size, args.overlap)
    else:
        store = PgVectorStore()
    return store, embedder
