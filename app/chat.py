from datetime import datetime
import re
from openai import OpenAI
from typing import Tuple

from .config import Config
from .utils import (
    load_db,
    save_db,
    ensure_user,
    parse_task_nl,
    parse_datetime_in_text,
    friendly_due,
    sentiment_bucket,
)


class ChatManager:
    def __init__(self, data_path=None):
        self.data_path = data_path or Config.DATA_PATH
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = Config.OPENAI_MODEL

    def _db(self):
        return load_db(self.data_path)

    def _save(self, data):
        save_db(self.data_path, data)

    # ---------- ANALISIS DE MENSAJES -------------

    def analyze_sentiment(self, text: str) -> Tuple[float, str]:
        prompt = (
            "Analiza este mensaje y devolvé score y label:\n"
            "score=<num entre -1 y 1>; label=<positivo|neutral|negativo>\n\n"
            f"Mensaje: {text}"
        )
        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Análisis breve de emociones."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            c = r.choices[0].message.content.lower()

            score = 0.0
            m = re.search(r"score\s*=\s*([-+]?\d+(\.\d+)?)", c)
            if m:
                score = float(m.group(1))

            if "positivo" in c:
                label = "positivo"
            elif "negativo" in c:
                label = "negativo"
            else:
                label = "neutral"

            return score, label

        except:
            return 0.0, "neutral"

    def coaching_reply(self, text: str, mood: str) -> str:
        msg = (
            "Respondé como un amigo que ayuda a organizarse. "
            "Usá tono humano, cálido, breve y concreto."
        )
        user_msg = f"Mensaje: {text}\nEmoción: {mood}"

        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.8,
            )
            return r.choices[0].message.content.strip()
        except:
            return "Tranqui, vamos paso a paso. Contame qué te queda por hacer."

    # ------------ TAREAS ------------------

    def add_task_from_text(self, user_id, text):
        db = self._db()
        user = ensure_user(db, user_id)

        parsed = parse_task_nl(text)
        parsed["title"] = parsed["title"].capitalize()
        parsed["created_at"] = datetime.now().isoformat()
        parsed["done"] = False

        user["tasks"].append(parsed)
        user["last_added_task"] = len(user["tasks"]) - 1

        self._save(db)

        return (
            f"✨ Listo, agendé: *{parsed['title']}*\n"
            f"Fecha: {friendly_due(parsed['due'])}\n"
            "¿Querés que lo recordemos de alguna manera especial?"
        )

    def edit_last_task_due(self, user_id, text):
        db = self._db()
        user = ensure_user(db, user_id)

        idx = user.get("last_added_task")
        if idx is None:
            return None

        dt = parse_datetime_in_text(text)
        if not dt:
            return None

        task = user["tasks"][idx]
        task["due"] = dt.isoformat()

        self._save(db)

        return (
            f"👌 De una, moví *{task['title']}* a {friendly_due(task['due'])}.\n"
            "¿Seguimos con algo más?"
        )

    def list_tasks(self, user_id, scope="pending"):
        db = self._db()
        user = ensure_user(db, user_id)

        tasks = user["tasks"]
        if scope == "pending":
            tasks = [t for t in tasks if not t["done"]]

        if not tasks:
            return "No tenés nada pendiente por ahora ✨"

        out = []
        for i, t in enumerate(tasks, 1):
            out.append(f"{i}. {'⬜' if not t['done'] else '✔️'} {t['title']} — {friendly_due(t['due'])}")
        return "\n".join(out)

    def mark_done(self, user_id, idx):
        db = self._db()
        user = ensure_user(db, user_id)

        pending = [t for t in user["tasks"] if not t["done"]]

        if idx < 1 or idx > len(pending):
            return "El número no es válido."

        task = pending[idx - 1]
        task["done"] = True

        self._save(db)

        return f"💛 Genial, marqué como hecha: *{task['title']}*"

    def mark_last_reminded_done(self, user_id):
        db = self._db()
        user = ensure_user(db, user_id)
        today = datetime.now().date().isoformat()

        tasks = [
            t for t in user["tasks"]
            if not t["done"] and t.get("last_reminder_date") == today
        ]

        if not tasks:
            return None

        t = tasks[-1]
        t["done"] = True
        self._save(db)

        return f"Perfecto 💛 marqué como completada *{t['title']}*.\n¿Querés que te recuerde otra cosa?"

    # ------------ RESUMEN + RECORDATORIOS ------------------

    def reflect_today(self, user_id):
        db = self._db()
        user = ensure_user(db, user_id)

        today = datetime.now().date().isoformat()
        moods = [m for m in user["moods"] if m["ts"][:10] == today]
        done = [t for t in user["tasks"] if t["done"]]
        pending = [t for t in user["tasks"] if not t["done"]]

        if moods:
            avg = sum(m["score"] for m in moods) / len(moods)
            mood_txt = sentiment_bucket(avg)
        else:
            mood_txt = "sin registro"

        return (
            "📋 *Resumen de hoy*\n"
            f"- Estado general: {mood_txt}\n"
            f"- Tareas hechas: {len(done)}\n"
            f"- Pendientes: {len(pending)}\n"
            "¿Querés que armemos un plan para seguir?"
        )

    def get_due_tasks_for_reminder(self):
        db = self._db()
        today = datetime.now().date().isoformat()

        out = {}

        for uid, user in db.get("users", {}).items():
            for t in user["tasks"]:
                if t["done"]:
                    continue
                if not t.get("due"):
                    continue

                try:
                    dt = datetime.fromisoformat(t["due"])
                except:
                    continue

                if dt.date().isoformat() != today:
                    continue

                if t.get("last_reminder_date") == today:
                    continue

                out.setdefault(uid, []).append(t)
                t["last_reminder_date"] = today

        self._save(db)
        return out
