#!/usr/bin/env python3
import sys

from rag.corpus import load_documents
from rag.generate import generate
from rag.pipeline import MiniRAG


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <вопрос>")
        raise SystemExit(1)
    question = " ".join(sys.argv[1:])

    rag = MiniRAG(load_documents())
    chunks = rag.retrieve(question)

    print("Найденные фрагменты:")
    for i, c in enumerate(chunks, 1):
        preview = c.text[:110].replace("\n", " ")
        print(f"  [{i}] ({c.label}) {preview}...")
    print()

    print("Ответ:")
    print(generate(question, chunks))


if __name__ == "__main__":
    main()
