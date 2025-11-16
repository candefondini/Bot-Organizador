from datetime import datetime, timedelta
import json
import re
from pathlib import Path
import dateparser


def sentiment_bucket(score: float) -> str:
    if score >= 0.25:
        return "positivo"
    if score <= -0.25:
        return "negativo"
    return "neutral"


PRIORITY_WORDS = {
    "urgente": 3,
    "importante": 2,
    "prioritario": 2,
    "alta": 3,
    "media": 2,
    "baja": 1,
}


def load_db(path: str) -> dict:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if not p.exists():
        with p.open("w", encoding="utf-8") as f:
            json.dump({"users": {}}, f, ensure_ascii=False, indent=2)

    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_db(path: str, data: dict) -> None:
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ensure_user(db: dict, user_id: str) -> dict:
    users = db.setdefault("users", {})
    return users.setdefault(
        user_id,
        {"tasks": [], "moods": [], "history": [], "last_added_task": None},
    )


def parse_datetime_in_text(text: str) -> datetime | None:
    lower = text.lower().strip()

    # 1) Intentar con dateparser (maneja "mañana a las 17", "viernes 10:30", etc.)
    dt = dateparser.parse(lower, languages=["es"])
    if dt:
        return dt

    # 2) "en un minuto / en 10 minutos / en 30 segundos"
    m = re.search(r"en\s+(un|una|\d+)\s+(segundos?|minutos?|minuto|segundo)", lower)
    if m:
        raw = m.group(1)
        unit = m.group(2)

        n = 1 if raw in ["un", "una"] else int(raw)

        if unit.startswith("seg"):
            return datetime.now() + timedelta(seconds=n)
        else:
            return datetime.now() + timedelta(minutes=n)

    # 3) "a las 17hs" / "a las 17 hs" / "a las 17:30"
    m = re.search(
        r"a las\s+(\d{1,2})(?:[:h](\d{2}))?\s*h?s?",
        lower,
    )
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        now = datetime.now()
        return now.replace(hour=hour, minute=minute, second=0, microsecond=0)

    return None


def parse_task_nl(text: str) -> dict:
    lower = text.lower().strip()

    priority = 1
    for w, p in PRIORITY_WORDS.items():
        if w in lower:
            priority = p

    dt = parse_datetime_in_text(lower)
    due_iso = dt.isoformat() if dt else None

    title = lower

    title = re.sub(
        r"\b(recordame|avisame|ponelo|agendalo|tengo que|debo|necesito que|movelo|pasalo)\b",
        "",
        title,
    )

    # sacar cosas tipo "a las 17hs", "a las 17:30", "a las 17"
    title = re.sub(r"\ba las\s+\d{1,2}(:\d{2})?\s*h?s?", "", title)
    # por si queda "17hs" suelto
    title = re.sub(r"\b\d{1,2}hs\b", "", title)

    title = title.strip(" ,.-")

    if not title:
        title = "Tarea"

    # capitalizar un poco
    title = title[0].upper() + title[1:] if title else title

    return {"title": title, "due": due_iso, "priority": priority}


def friendly_due(due_iso: str | None) -> str:
    if not due_iso:
        return "sin fecha"
    try:
        dt = datetime.fromisoformat(due_iso)
        return dt.strftime("%d/%m %H:%M")
    except Exception:
        return "sin fecha"
