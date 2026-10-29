"""Mede recall@3 e MRR da busca e testa a recusa por limiar.

    python eval/run_eval.py --store memory --embedder hash
    python eval/run_eval.py --store pgvector            (depois de python ingest.py)

Definições (nível de trecho, não de seção):
  recall@3 = fração das perguntas respondíveis em que algum dos 3 primeiros trechos
             vem da seção esperada.
  MRR      = média de 1/posição do primeiro trecho da seção esperada (olhando os 10
             primeiros; 0 se não aparece).
  recusa   = a pergunta é recusada quando o score do melhor trecho < limiar.
             Perguntas sem resposta no corpus (expected = null) deveriam ser recusadas;
             as respondíveis não.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.pipeline import add_common_args, open_index, setup_stdout  # noqa: E402


def evaluate(questions, store, embedder, min_score, k=3):
    rows = []
    for q in questions:
        hits = store.search(embedder.embed([q["question"]])[0], 10)
        top = hits[0].score if hits else 0.0
        row = {"id": q["id"], "question": q["question"], "expected": q["expected"],
               "top_score": top, "refused": not top >= min_score, "rank": None,
               "top_sections": [h.source for h in hits[:k]]}
        if q["expected"]:
            ranks = [i for i, h in enumerate(hits, 1) if h.source == q["expected"]]
            row["rank"] = ranks[0] if ranks else None
        rows.append(row)

    answerable = [r for r in rows if r["expected"]]
    unanswerable = [r for r in rows if not r["expected"]]
    return {
        "rows": rows,
        "n_answerable": len(answerable),
        "recall_at_k": sum(1 for r in answerable if r["rank"] and r["rank"] <= k) / len(answerable),
        "mrr": sum(1 / r["rank"] for r in answerable if r["rank"]) / len(answerable),
        "false_refusals": sum(1 for r in answerable if r["refused"]),
        "n_unanswerable": len(unanswerable),
        "correct_refusals": sum(1 for r in unanswerable if r["refused"]),
        "k": k,
    }


def main():
    setup_stdout()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--questions", default=str(Path(__file__).with_name("questions.json")))
    parser.add_argument("--min-score", type=float, default=None)
    add_common_args(parser)
    args = parser.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    store, embedder = open_index(args)
    min_score = embedder.min_score if args.min_score is None else args.min_score
    r = evaluate(questions, store, embedder, min_score)

    print(f"embedder={embedder.name} store={args.store} limiar={min_score}\n")
    print(f"{'id':4} {'rank':>4} {'top1':>6}  {'recusou':7}  pergunta")
    for row in r["rows"]:
        rank = "-" if row["expected"] is None else (row["rank"] or ">10")
        print(f"{row['id']:4} {rank!s:>4} {row['top_score']:6.3f}  {'sim' if row['refused'] else 'não':7}  {row['question']}")
    k = r["k"]
    print(f"\nrecall@{k}: {r['recall_at_k']:.3f}  ({round(r['recall_at_k'] * r['n_answerable'])}/{r['n_answerable']})")
    print(f"MRR:      {r['mrr']:.3f}")
    print(f"recusas corretas (sem resposta no corpus): {r['correct_refusals']}/{r['n_unanswerable']}")
    print(f"falsas recusas (com resposta no corpus):   {r['false_refusals']}/{r['n_answerable']}")


if __name__ == "__main__":
    main()
