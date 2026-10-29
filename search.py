"""Busca os k trechos mais parecidos com a pergunta, com a pontuação de cada um.

    python search.py "Quantos sacos de cimento posso empilhar?" -k 3
"""
import argparse

from rag.pipeline import add_common_args, open_index, setup_stdout


def main():
    setup_stdout()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=3)
    add_common_args(parser)
    args = parser.parse_args()

    store, embedder = open_index(args)
    for rank, h in enumerate(store.search(embedder.embed([args.query])[0], args.k), 1):
        snippet = h.text if len(h.text) <= 220 else h.text[:217] + "..."
        print(f"{rank}. score={h.score:.3f}  {h.section}  ({h.source}#{h.index})\n   {snippet}")


if __name__ == "__main__":
    main()
