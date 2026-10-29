"""Testa a lógica de `evaluate()` (a que gera os números do README) com um store de ranking conhecido."""
import importlib.util
from types import SimpleNamespace

from rag.pipeline import ROOT

spec = importlib.util.spec_from_file_location("run_eval", ROOT / "eval" / "run_eval.py")
run_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_eval)


def hit(source, score):
    return SimpleNamespace(source=source, score=score)


class FakeEmbedder:
    def embed(self, texts):
        return [[t] for t in texts]  # o "vetor" é a própria pergunta


class FakeStore:
    """Devolve um ranking fixo por pergunta, já ordenado do melhor para o pior."""

    def __init__(self, rankings):
        self.rankings = rankings

    def search(self, vector, k=3):
        return self.rankings[vector[0]][:k]


def test_evaluate_calcula_recall_mrr_e_recusas():
    questions = [
        {"id": "a", "question": "a", "expected": "A.md"},  # certo em 1º
        {"id": "b", "question": "b", "expected": "B.md"},  # certo em 2º
        {"id": "d", "question": "d", "expected": "D.md"},  # certo em 3º: borda do recall@3
        {"id": "c", "question": "c", "expected": "C.md"},  # ausente
        {"id": "x", "question": "x", "expected": None},    # sem resposta, score baixo: recusa certa
        {"id": "y", "question": "y", "expected": None},    # sem resposta, score alto: não recusa
    ]
    store = FakeStore({
        "a": [hit("A.md", 0.9), hit("Z.md", 0.5)],
        "b": [hit("Z.md", 0.8), hit("B.md", 0.4)],
        "d": [hit("Z.md", 0.8), hit("Z.md", 0.7), hit("D.md", 0.6)],
        "c": [hit("Z.md", 0.7), hit("Y.md", 0.6)],
        "x": [hit("Z.md", 0.1)],
        "y": [hit("Z.md", 0.6)],
    })
    r = run_eval.evaluate(questions, store, FakeEmbedder(), min_score=0.3)

    assert r["recall_at_k"] == 3 / 4  # a, b e d (rank 3 conta); c não
    assert r["mrr"] == (1 + 0.5 + 1 / 3 + 0) / 4
    assert (r["n_answerable"], r["n_unanswerable"]) == (4, 2)
    assert r["correct_refusals"] == 1
    assert r["false_refusals"] == 0
    assert [row["rank"] for row in r["rows"]] == [1, 2, 3, None, None, None]
    assert [row["refused"] for row in r["rows"]] == [False, False, False, False, True, False]


def test_evaluate_recall_respeita_o_k_e_conta_falsa_recusa():
    # Seção certa em 4º: fora do top-3 (recall 0), mas conta no MRR (1/4).
    hits = [hit("Z.md", 0.9), hit("Z.md", 0.8), hit("Z.md", 0.7), hit("A.md", 0.6)]
    questions = [{"id": "a", "question": "a", "expected": "A.md"}]
    r = run_eval.evaluate(questions, FakeStore({"a": hits}), FakeEmbedder(), min_score=0.95)

    assert r["recall_at_k"] == 0
    assert r["mrr"] == 0.25
    assert r["false_refusals"] == 1  # melhor score 0,9 < limiar 0,95 numa pergunta respondível


def test_evaluate_score_nan_conta_como_recusa():
    questions = [
        {"id": "a", "question": "a", "expected": "A.md"},
        {"id": "x", "question": "x", "expected": None},
    ]
    store = FakeStore({"a": [hit("A.md", 0.9)], "x": [hit("Z.md", float("nan"))]})
    r = run_eval.evaluate(questions, store, FakeEmbedder(), min_score=0.3)

    assert [row["refused"] for row in r["rows"]] == [False, True]
    assert r["correct_refusals"] == 1
