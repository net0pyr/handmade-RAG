import requests

PROMPT = """Ты отвечаешь ТОЛЬКО на основе фрагментов ниже.
Если ответа в них нет — напиши: «В базе знаний нет ответа на этот вопрос».
После каждого утверждения ставь номер фрагмента в квадратных скобках.

<фрагменты>
{context}
</фрагменты>

Вопрос: {question}"""

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:8b"


def generate(question: str, chunks: list[str]) -> str:
    context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(chunks))
    r = requests.post(OLLAMA_URL, timeout=120, json={
        "model": MODEL,
        "prompt": PROMPT.format(context=context, question=question),
        "stream": False,
        "options": {"temperature": 0.1},
    })
    r.raise_for_status()
    return r.json()["response"]
