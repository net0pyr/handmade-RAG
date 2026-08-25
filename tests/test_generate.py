from rag.chunking import Chunk
from rag.generate import build_context


def test_context_carries_the_source_into_the_prompt():
    """Ссылка «[3]» без имени документа непроверяема, ради этого метаданные и тащатся."""
    chunks = [Chunk("текст про VPN", "vpn.md", 0), Chunk("текст про отпуск", "vacation.md", 2)]
    context = build_context(chunks)
    assert "[1] (vpn.md, фрагмент 1)" in context
    assert "[2] (vacation.md, фрагмент 3)" in context
