import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POLICIES_DIR = ROOT / "data" / "policies"
GOLDEN_PATH = ROOT / "data" / "golden.json"


def load_documents(directory: Path = POLICIES_DIR) -> dict[str, str]:
    """Имя файла -> текст. Имя нужно, чтобы ответ можно было проверить по источнику."""
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(directory.glob("*.md"))}


def load_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    """Вопросы с разметкой «какой документ считается правильным ответом»."""
    return json.loads(path.read_text(encoding="utf-8"))
