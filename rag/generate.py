import os
import re

import requests

from .chunking import Chunk

PROMPT = """Ты отвечаешь ТОЛЬКО на основе фрагментов ниже.
Если ответа в них нет — напиши: «В базе знаний нет ответа на этот вопрос».
После каждого утверждения ставь номер фрагмента в квадратных скобках.

<фрагменты>
{context}
</фрагменты>

Вопрос: {question}"""

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.getenv("RAG_MODEL", "qwen3:8b")

# Ollama по умолчанию поднимает окно всего в 4096 токенов независимо от того,
# сколько заявлено у модели, и молча обрезает то, что не влезло. Для RAG это
# худший из возможных отказов: ошибки нет, просто часть контекста не доехала.
NUM_CTX = int(os.getenv("RAG_NUM_CTX", "8192"))

# Ключ think отправляем, только если его задали явно. Пустое значение — не
# отправляем вовсе, и это осознанно: на Ollama 0.32 с qwen3 "think": false не
# выключает рассуждения, а лишь отменяет их отделение от ответа — рассуждения
# приезжают прямо в response вместе с висящим тегом </think>.
THINK = os.getenv("RAG_THINK")

TIMEOUT = int(os.getenv("RAG_TIMEOUT", "300"))

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Грубая оценка: для русского текста примерно 3 символа на токен. Специально
# занижено, чтобы предупреждение срабатывало раньше, чем Ollama начнёт резать.
CHARS_PER_TOKEN = 3


def build_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{i}] ({c.label})\n{c.text}" for i, c in enumerate(chunks, 1))


def strip_thinking(text: str) -> str:
    """Убираем рассуждения, если они всё-таки просочились в ответ."""
    text = _THINK_BLOCK_RE.sub("", text)
    _, tag, tail = text.rpartition("</think>")   # бывает и без открывающего тега
    return (tail if tag else text).strip()


def generate(question: str, chunks: list[Chunk]) -> str:
    prompt = PROMPT.format(context=build_context(chunks), question=question)

    estimated = len(prompt) // CHARS_PER_TOKEN
    if estimated > NUM_CTX:
        raise ValueError(
            f"промпт ~{estimated} токенов при окне {NUM_CTX}: Ollama обрежет его молча. "
            f"Уменьшите k или поднимите RAG_NUM_CTX."
        )

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": NUM_CTX},
    }
    if THINK is not None:
        payload["think"] = THINK == "1"

    try:
        r = requests.post(f"{OLLAMA_HOST}/api/generate", timeout=TIMEOUT, json=payload)
    except requests.ConnectionError as e:
        raise RuntimeError(
            f"Ollama недоступна на {OLLAMA_HOST}. Запустите `ollama serve`."
        ) from e
    r.raise_for_status()
    return strip_thinking(r.json()["response"])
