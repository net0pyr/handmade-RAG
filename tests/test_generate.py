from rag.chunking import Chunk
from rag.generate import build_context, strip_thinking


def test_context_carries_the_source_into_the_prompt():
    chunks = [Chunk("текст про VPN", "vpn.md", 0), Chunk("текст про отпуск", "vacation.md", 2)]
    context = build_context(chunks)
    assert "[1] (vpn.md, фрагмент 1)" in context
    assert "[2] (vacation.md, фрагмент 3)" in context


def test_strip_thinking_removes_a_full_block():
    assert strip_thinking("<think>рассуждения</think>\nОтвет [1]") == "Ответ [1]"


def test_strip_thinking_handles_missing_opening_tag():
    """Ollama срезает открывающий тег, и в ответе остаётся только закрывающий."""
    assert strip_thinking("Хм, надо подумать...\n</think>\n\nОтвет [1]") == "Ответ [1]"


def test_strip_thinking_leaves_a_clean_answer_alone():
    assert strip_thinking("  Ответ [1]  ") == "Ответ [1]"
