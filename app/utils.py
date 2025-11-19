# app/utils.py
# Este archivo contiene funciones auxiliares que usa todo el bot:
# - Interpretación de fechas en lenguaje natural
# - Manejo de la “base de datos” JSON
# - Limpieza y parseo básico de tareas
# - Categorización de sentimientos
# - Utilidades generales de formato

from datetime import datetime, timedelta
import json
import re
from pathlib import Path
import dateparser


def sentiment_bucket(score: float) -> str:
    # Clasifica un valor numérico en positivo / neutral / negativo.
    # Se usa para resumir el estado emocional del usuario.
    if score >= 0.25:
        return "positivo"
    if score <= -0.25:
        return "negativo"
    return "neutral"


# Palabras que determinan prioridad cuando GPT falla.
# Por ejemplo: “urgente” → prioridad 3.
PRIORITY_WORDS = {
    "urgente": 3,
    "importante": 2,
    "prioritario": 2,
    "alta": 3,
    "media": 2,
    "baja": 1,
}


def load_db(path: str) -> dict:
    # Carga el archivo JSON donde se guarda la información de los usuarios.
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Si el archivo no existe, lo crea con la estructura inicial.
    if not p.exists():
        with p.open("w", encoding="utf-8") as f:
            json.dump({"users": {}}, f, ensure_ascii=False, indent=2)

    # Devuelve el diccionario del archivo JSON.
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_db(path: str, data: dict) -> None:
    # Guarda el diccionario de datos en el archivo JSON.
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ensure_user(db: dict, user_id: str) -> dict:
    # Garantiza que el usuario exista dentro del archivo JSON.
    # Si no existe, crea la estructura por defecto.
    users = db.setdefault("users", {})

    default_user = {
        "tasks": [],
        "reminders": [],          
        "moods": [],
        "history": [],
        "last_added_task": None,
    }

    # Obtiene o crea la entrada del usuario.
    user = users.setdefault(user_id, default_user)

    # Compatibilidad con versiones viejas del bot que no tenían reminders.
    if "reminders" not in user:
        user["reminders"] = []

    return user


def parse_datetime_in_text(text: str) -> datetime | None:
    """
    Convierte expresiones de tiempo del tipo:
    - "en 5 minutos"
    - "a las 18"
    - "mañana a las 10"
    - "viernes 14:30"
    - "en un minuto"
    
    Devuelve SIEMPRE una fecha/hora FUTURA.
    Si algo no se puede interpretar, devuelve None.
    """
    lower = text.lower().strip()
    now = datetime.now()

    # 1) Expresiones del tipo "en X minutos/horas/segundos"
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

    # 2) Expresiones del tipo "a las 17", "a las 17:30", con o sin 'hs'
    m = re.search(
        r"a las\s+(\d{1,2})(?:[:h](\d{2}))?\s*h?s?",
        lower,
    )
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        # Si la hora ya pasó hoy, lo agenda para mañana.
        if target <= now:
            target = target + timedelta(days=1)
        
        return target

    # 3) Como último recurso, usar dateparser para entender frases más complejas.
    dt = dateparser.parse(
        lower, 
        languages=["es"],
        settings={
            'PREFER_DATES_FROM': 'future',  # Siempre preferir futuro
            'RELATIVE_BASE': now
        }
    )
    
    if dt:
        # Ajuste por si dateparser devolvió una hora pasada
        if dt <= now:
            if "mañana" not in lower and "ayer" not in lower and any(x in lower for x in ["a las", ":", "hs"]):
                dt = dt + timedelta(days=1)
        
        return dt

    # Si no se pudo interpretar nada
    return None


def parse_task_nl(text: str) -> dict:
    """
    Analizador básico de tareas cuando GPT falla.
    - Detecta prioridad por palabras
    - Limpia expresiones como "recordame", "tengo que"
    - Devuelve un título simple y una prioridad
    """
    lower = text.lower().strip()

    priority = 1
    # Busca palabras que indiquen prioridad
    for w, p in PRIORITY_WORDS.items():
        if w in lower:
            priority = p

    title = lower

    # Quitamos palabras típicas que no aportan al título
    title = re.sub(
        r"\b(recordame|avisame|ponelo|agendalo|tengo que|debo|necesito que|hoy)\b",
        "",
        title,
    )

    title = title.strip(" ,.-")

    if not title:
        title = "Tarea"

    # Capitalizar primera letra
    title = title[0].upper() + title[1:] if title else title

    return {"title": title, "priority": priority}


def friendly_due(due_iso: str | None) -> str:
    """Convierte una fecha ISO (2025-11-16T17:30:00) en formato legible: DD/MM HH:MM"""
    if not due_iso:
        return "sin fecha"
    try:
        dt = datetime.fromisoformat(due_iso)
        return dt.strftime("%d/%m %H:%M")
    except Exception:
        return "sin fecha"
