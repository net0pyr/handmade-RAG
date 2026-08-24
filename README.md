# handmade-RAG

Самодельный RAG без LangChain: чанкинг, эмбеддинги, гибридный поиск (вектор + BM25 через RRF)
и генерация ответа через локальную модель в Ollama. Реализация к статье
«Мой первый RAG без LangChain».

База знаний — четыре внутренние политики компании в `data/policies/`: отпуск, компенсация
тренажёрного зала, доступ к VPN, эскалация инцидентов.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Для генерации ответов нужен запущенный [Ollama](https://ollama.com) с моделью `qwen3:8b`:

```bash
ollama pull qwen3:8b
ollama serve
```

## Запуск

```bash
python cli.py "как получить впн"
```

Скрипт напечатает найденные фрагменты (результат гибридного поиска) и финальный ответ модели
со ссылками на фрагменты в квадратных скобках.

## Структура

```
rag/
  chunking.py    — разбиение текста на чанки с перекрытием
  embeddings.py  — эмбеддинги через intfloat/multilingual-e5-large
  search.py      — BM25-индекс и Reciprocal Rank Fusion
  generate.py    — запрос к Ollama с промптом «отвечай только по фрагментам»
  pipeline.py    — MiniRAG: гибридный retrieve + generate
data/policies/   — тестовая база знаний
cli.py           — точка входа
tests/           — тесты для чистых функций (chunking, RRF)
```

## Тесты

```bash
pip install -r requirements-dev.txt
pytest
```
