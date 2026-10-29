from types import SimpleNamespace

from rag.prompt import REFUSAL, SYSTEM, answer, build_prompt
from rag.stores import Hit


def hit(score, n=1):
    return Hit(score, f"{n:02d}-x.md", f"Seção {n}", 0, f"texto da fonte {n}")


class FakeClient:
    """Imita client.messages.create sem rede."""

    def __init__(self, text="Resposta [1].", stop_reason="end_turn"):
        self.calls = []
        self.messages = self
        self._resp = SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)])

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._resp


def test_prompt_numera_fontes_e_tem_regra_de_recusa():
    system, user = build_prompt("Qual a altura?", [hit(0.9, 1), hit(0.8, 2)])
    assert "[1] (Seção: Seção 1" in user and "[2] (Seção: Seção 2" in user
    assert user.endswith("Pergunta: Qual a altura?")
    assert REFUSAL in system and "SOMENTE" in system
    assert "[n]" in system and "exatamente" in system  # regra de citação e de recusa literal
    assert "dado, não instrução" in system  # texto das fontes não vira ordem (injeção de prompt)


def test_abaixo_do_limiar_recusa_sem_chamar_o_modelo():
    client = FakeClient()
    r = answer("pergunta", [hit(0.10)], min_score=0.30, client=client)
    assert (r.mode, r.text) == ("recusa", REFUSAL)
    assert client.calls == []


def test_score_nan_recusa_sem_chamar_o_modelo():
    client = FakeClient()
    r = answer("pergunta", [hit(float("nan"))], min_score=0.30, client=client)
    assert r.mode == "recusa" and client.calls == []


def test_sem_cliente_roda_extrativo():
    r = answer("pergunta", [hit(0.9, 1), hit(0.8, 2)], min_score=0.30)
    assert r.mode == "extrativo"
    assert "[1] Seção 1" in r.text and "texto da fonte 2" in r.text


def test_com_cliente_envia_prompt_e_devolve_texto():
    client = FakeClient("A altura é 1,20 m [1].")
    r = answer("Qual a altura?", [hit(0.9)], min_score=0.30, client=client, model="modelo-teste")
    call = client.calls[0]
    assert r.mode == "llm" and r.text == "A altura é 1,20 m [1]."
    assert call["model"] == "modelo-teste" and call["system"] == SYSTEM
    assert "texto da fonte 1" in call["messages"][0]["content"]


def test_recusa_do_modelo_vira_mensagem_clara():
    r = answer("pergunta", [hit(0.9)], min_score=0.30, client=FakeClient(stop_reason="refusal"))
    assert r.mode == "recusa"
