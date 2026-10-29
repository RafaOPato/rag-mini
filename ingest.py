"""Lê o corpus, divide em trechos, gera os vetores e grava no store.

    python ingest.py                       # pgvector (precisa do docker compose up)
    python ingest.py --store memory        # só mostra os trechos; memory não persiste
"""
import argparse

from rag.chunking import chunk_corpus, overlap_boundaries
from rag.embedders import get_embedder
from rag.pipeline import add_common_args, ingest, setup_stdout
from rag.stores import MemoryStore, PgVectorStore


def main():
    setup_stdout()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    add_common_args(parser)
    parser.add_argument("-v", "--verbose", action="store_true", help="lista os trechos")
    args = parser.parse_args()

    embedder = get_embedder(args.embedder)
    store = MemoryStore() if args.store == "memory" else PgVectorStore()
    n = ingest(store, embedder, args.corpus, args.size, args.overlap)
    sections = len({c.source for c in chunk_corpus(args.corpus, args.size, args.overlap)})
    print(f"{n} trechos de {sections} seções indexados "
          f"(embedder={embedder.name}, dim={embedder.dim}, store={args.store}).")
    if args.store == "memory":
        print("Obs.: o store 'memory' não persiste; search.py e ask.py reindexam a cada execução.")
    if args.verbose:
        repeated, total = overlap_boundaries(args.corpus, args.size, args.overlap)
        print(f"fronteiras entre trechos que repetem texto do anterior: {repeated}/{total}")
        for c in chunk_corpus(args.corpus, args.size, args.overlap):
            print(f"- {c.source}#{c.index} ({len(c.text)} caracteres)")


if __name__ == "__main__":
    main()
