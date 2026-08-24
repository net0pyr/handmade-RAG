#!/usr/bin/env python3
import sys
from pathlib import Path

from rag.generate import generate
from rag.pipeline import MiniRAG

DATA_DIR = Path(__file__).parent / "data" / "policies"


def load_documents() -> list[str]:
    return [p.read_text(encoding="utf-8") for p in sorted(DATA_DIR.glob("*.md"))]


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <вопрос>")
        raise SystemExit(1)
    question = " ".join(sys.argv[1:])

    rag = MiniRAG(load_documents())
    chunks = rag.retrieve(question)

    print("Найденные фрагменты:")
    for i, c in enumerate(chunks, 1):
        preview = c[:120].replace("\n", " ")
        print(f"  [{i}] {preview}...")
    print()

    print("Ответ:")
    print(generate(question, chunks))


if __name__ == "__main__":
    main()
