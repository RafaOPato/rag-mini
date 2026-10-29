import importlib.util
import json
import os

import pytest

from rag.embedders import HashEmbedder
from rag.pipeline import ROOT, ingest
from rag.stores import MemoryStore


def load_run_eval():
    spec = importlib.util.spec_from_file_location("run_eval", ROOT / "eval" / "run_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


QUESTIONS = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))


def test_perguntas_apontam_para_secoes_que_existem():
    files = {p.name for p in (ROOT / "corpus").glob("*.md")}
    assert {q["expected"] for q in QUESTIONS if q["expected"]} <= files
    assert 12 <= sum(1 for q in QUESTIONS if q["expected"]) <= 15
    assert 2 <= sum(1 for q in QUESTIONS if not q["expected"]) <= 3


def test_eval_com_hash_nao_regride():
    # Piso de regressão, não meta: o número real está no README.
    e, store = HashEmbedder(), MemoryStore()
    ingest(store, e, ROOT / "corpus")
    r = load_run_eval().evaluate(QUESTIONS, store, e, e.min_score)
    assert r["recall_at_k"] >= 0.7
    assert r["mrr"] >= 0.5


@pytest.mark.skipif(not os.environ.get("RAG_TEST_PG"), reason="defina RAG_TEST_PG=1 com o docker compose de pé")
def test_pgvector_da_o_mesmo_resultado_que_memory():
    from rag.stores import PgVectorStore

    e = HashEmbedder()
    pg, mem = PgVectorStore(table="chunks_test"), MemoryStore()
    try:
        ingest(pg, e, ROOT / "corpus")
        ingest(mem, e, ROOT / "corpus")
        for q in QUESTIONS:
            v = e.embed([q["question"]])[0]
            a, b = pg.search(v, 3), mem.search(v, 3)
            assert [(h.source, h.index) for h in a] == [(h.source, h.index) for h in b]
            assert all(abs(x.score - y.score) < 1e-4 for x, y in zip(a, b))
        # pergunta só de stopwords = vetor zero: o `<=>` do pgvector dá NaN, e o store tem que devolver 0.0 como o memory
        assert [h.score for h in pg.search([0.0] * e.dim, 3)] == [0.0] * 3
    finally:
        pg.conn.execute("DROP TABLE IF EXISTS chunks_test")
