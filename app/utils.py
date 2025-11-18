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

    # Valores por defecto del usuario
    default_user = {
        "tasks": [],
        "reminders": [],          
        "moods": [],
        "history": [],
        "last_added_task": None,
    }

    user = users.setdefault(user_id, default_user)

    # Por si hay usuarios viejos guardados sin 'reminders', lo agregamos igual
    if "reminders" not in user:
        user["reminders"] = []

    return user


def parse_datetime_in_text(text: str) -> datetime | None:
    """
    Parsea expresiones de tiempo en texto natural.
    SIEMPRE devuelve fechas/horas FUTURAS.
    """
    lower = text.lower().strip()
    now = datetime.now()

    # 1) "en un minuto / en 10 minutos / en 30 segundos / en 2 horas"
    m = re.search(r"en\s+(un|una|\d+)\s+(segundos?|minutos?|minuto|segundo|horas?|hora)", lower)
    if m:
        raw = m.group(1)
        unit = m.group(2)

        n = 1 if raw in ["un", "una"] else int(raw)

        if unit.startswith("seg"):
            return now + timedelta(seconds=n)
        elif unit.startswith("min"):
            return now + timedelta(minutes=n)
        elif unit.startswith("hora"):
            return now + timedelta(hours=n)

    # 2) "a las 17hs" / "a las 17 hs" / "a las 17:30"
    m = re.search(
        r"a las\s+(\d{1,2})(?:[:h](\d{2}))?\s*h?s?",
        lower,
    )
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        # Si la hora ya pasó hoy, ponerla para mañana
        if target <= now:
            target = target + timedelta(days=1)
        
        return target

    # 3) Intentar con dateparser (maneja "mañana a las 17", "viernes 10:30", etc.)
    dt = dateparser.parse(
        lower, 
        languages=["es"],
        settings={
            'PREFER_DATES_FROM': 'future',  # Preferir fechas futuras
            'RELATIVE_BASE': now
        }
    )
    
    if dt:
        # Si dateparser devolvió una fecha pasada, ajustarla
        if dt <= now:
            # Si es solo hora sin fecha explícita, moverla al día siguiente
            if "mañana" not in lower and "ayer" not in lower and any(x in lower for x in ["a las", ":", "hs"]):
                dt = dt + timedelta(days=1)
        
        return dt

    return None


def parse_task_nl(text: str) -> dict:
    """
    Fallback simple para parsear tareas cuando GPT falla.
    Las tareas ya NO tienen fecha (son del día actual).
    """
    lower = text.lower().strip()

    priority = 1
    for w, p in PRIORITY_WORDS.items():
        if w in lower:
            priority = p

    title = lower

    # Limpiar palabras innecesarias
    title = re.sub(
        r"\b(recordame|avisame|ponelo|agendalo|tengo que|debo|necesito que|hoy)\b",
        "",
        title,
    )

    title = title.strip(" ,.-")

    if not title:
        title = "Tarea"

    # Capitalizar un poco
    title = title[0].upper() + title[1:] if title else title

    return {"title": title, "priority": priority}


def friendly_due(due_iso: str | None) -> str:
    """Convierte un datetime ISO a formato legible"""
    if not due_iso:
        return "sin fecha"
    try:
        dt = datetime.fromisoformat(due_iso)
        return dt.strftime("%d/%m %H:%M")
    except Exception:
        return "sin fecha"