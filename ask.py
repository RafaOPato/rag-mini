"""Responde uma pergunta usando só as fontes recuperadas.

    python ask.py "Em que profundidade a vala precisa de escoramento?"

Com ANTHROPIC_API_KEY definida, chama o Claude (SDK oficial `anthropic`).
Sem a chave (ou com --no-llm), roda no modo extrativo: imprime os trechos recuperados.
Se o melhor trecho tiver score abaixo do limiar, recusa sem chamar modelo nenhum.
"""
import argparse
import os

from rag.pipeline import add_common_args, open_index, setup_stdout
from rag.prompt import answer


def main():
    setup_stdout()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=None, help="limiar de recusa (padrão: o do embedder)")
    parser.add_argument("--no-llm", action="store_true", help="força o modo extrativo")
    add_common_args(parser)
    args = parser.parse_args()

    store, embedder = open_index(args)
    hits = store.search(embedder.embed([args.question])[0], args.k)

    client = None
    if os.environ.get("ANTHROPIC_API_KEY") and not args.no_llm:
        import anthropic

        client = anthropic.Anthropic()  # lê ANTHROPIC_API_KEY do ambiente

    min_score = embedder.min_score if args.min_score is None else args.min_score
    result = answer(args.question, hits, min_score, client)
    best = f"{hits[0].score:.3f}" if hits else "n/a"
    print(f"[modo: {result.mode} | melhor score: {best} | limiar: {min_score}]\n")
    print(result.text)


if __name__ == "__main__":
    main()
