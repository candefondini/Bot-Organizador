from datetime import datetime, timedelta
import re
from openai import OpenAI
from typing import Tuple, Optional, Dict, List
import json

from .config import Config
from .utils import (
    load_db,
    save_db,
    ensure_user,
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

    def _get_conversation_context(self, user_id: str, limit: int = 5) -> List[Dict]:
        """Obtiene el contexto de conversación reciente"""
        db = self._db()
        user = ensure_user(db, user_id)
        return user.get("history", [])[-limit:]

    # ---------- CLASIFICACIÓN INTELIGENTE DE INTENCIÓN -------------

    def classify_intent(self, text: str, context: List[Dict]) -> Dict:
        """
        Usa GPT para entender QUÉ quiere hacer el usuario.
        """
        context_str = "\n".join([
            f"- {h.get('type', 'msg')}: {h.get('raw', '')[:100]}"
            for h in context[-3:]
        ]) if context else "Sin contexto previo"

        prompt = f"""Analiza este mensaje de un usuario y determina su intención principal.

Contexto reciente:
{context_str}

Mensaje actual: "{text}"

Devolve SOLO un JSON con esta estructura:
{{
  "intent": "<una de: create_task, create_reminder, query_tasks, query_reminders, mark_done, mark_all_done, delete_task, delete_reminder, modify_reminder, chat, express_emotion, query_stats>",
  "confidence": <0.0 a 1.0>,
  "extracted_data": {{
    "task_title": "<si es tarea, el título limpio>",
    "datetime": "<cualquier fecha/hora mencionada>",
    "task_reference": "<si menciona 'la última', 'esa', 'todas', 'todo', etc>",
    "mark_all": <true si quiere marcar TODAS las tareas como hechas>,
    "emotion": "<si expresa emoción, cuál>",
    "query_scope": "<si pregunta por tareas: 'today', 'all', 'completed', etc>"
  }}
}}

Intenciones:
- create_task: quiere crear una TAREA del día (ej: "hoy tengo que comprar pan", "limpiar mi pieza")
- create_reminder: quiere un RECORDATORIO en momento específico (ej: "recordame llamar a Juan en 5 minutos", "avisame a las 15hs reunión")
- query_tasks: pregunta qué tareas tiene (para hoy)
- query_reminders: pregunta por recordatorios programados
- query_stats: pregunta por estadísticas, productividad, tareas completadas
- mark_done: indica que terminó UNA tarea específica
- mark_all_done: indica que terminó TODAS las tareas (ej: "marcalas todas", "hice todo")
- delete_task: quiere eliminar una tarea
- delete_reminder: quiere eliminar un recordatorio
- modify_reminder: quiere cambiar/mover un recordatorio
- chat: conversación casual
- express_emotion: expresa cómo se siente

REGLAS CRÍTICAS:
- Si dice "recordame", "avisame", "acordate de" + tiempo específico → create_reminder
- Si dice tareas para "hoy" sin pedir aviso → create_task
- Si pregunta por "completadas", "terminadas", "estadísticas" → query_stats
- Si pregunta por "recordatorios" → query_reminders
- Si quiere "borrar recordatorio" → delete_reminder
- Si quiere "mover/reagendar recordatorio" → modify_reminder

Ejemplos:
"hoy tengo que: 1- comprar pan 2- estudiar" → create_task (múltiples)
"recordame llamar en 5 minutos" → create_reminder
"qué tengo para hoy?" → query_tasks
"qué recordatorios tengo?" → query_reminders
"cuántas tareas hice hoy?" → query_stats
"ya lo hice" → mark_done
"hice todo" → mark_all_done
"borrá el recordatorio de la reunión" → delete_reminder
"cambiá el recordatorio del turno al oculista de mañana a las 15hs a pasado mañana a las 10hs" → modify_reminder
"""

        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Sos un clasificador de intenciones. Respondé SOLO con JSON válido."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )

            content = r.choices[0].message.content.strip()
            content = content.replace("```json", "").replace("```", "").strip()
            result = json.loads(content)

            if "intent" not in result:
                result = {
                    "intent": "chat",
                    "confidence": 0.5,
                    "extracted_data": {},
                }

            return result

        except Exception as e:
            print(f"Error en classify_intent: {e}")
            return {
                "intent": "chat",
                "confidence": 0.3,
                "extracted_data": {},
            }

    # ---------- EXTRACCIÓN INTELIGENTE DE RECORDATORIOS -------------

    def extract_reminder_smart(self, text: str) -> Dict:
        """
        Extrae información de un recordatorio usando GPT.
        """
        prompt = f"""Extraé la información de este recordatorio:

"{text}"

Devolve SOLO un JSON:
{{
  "title": "<qué debe recordar>",
  "time_expression": "<la expresión temporal EXACTA del mensaje: 'en 5 minutos', 'a las 15:30', 'mañana 10am', etc>",
  "notes": "<contexto adicional si hay>"
}}

Reglas:
- El título debe ser claro y accionable
- NO incluyas "recordame" en el título
- La time_expression debe ser EXACTAMENTE como apareció en el mensaje original
"""

        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Extraés recordatorios de texto natural. Respondé SOLO JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )

            content = r.choices[0].message.content.strip()
            content = content.replace("```json", "").replace("```", "").strip()
            result = json.loads(content)

            return result

        except Exception as e:
            print(f"Error en extract_reminder_smart: {e}")
            return {
                "title": text,
                "time_expression": None,
                "notes": ""
            }

    # ---------- EXTRACCIÓN INTELIGENTE DE TAREAS -------------

    def extract_task_smart(self, text: str, intent_data: Dict) -> Dict:
        """
        Usa GPT para extraer una tarea estructurada del texto natural.
        """
        prompt = f"""Extraé la información de esta tarea:

"{text}"

Devolve SOLO un JSON:
{{
  "title": "<título claro y conciso de la tarea>",
  "priority": <1-3, donde 3=urgente>,
  "notes": "<detalles adicionales si hay>"
}}

Reglas:
- El título debe ser accionable y claro
- NO incluyas palabras como "recordame", "avisame", "hoy tengo que"
- Preserva la esencia pero limpialo
- Si dice "urgente" o "ya", priority=3
- Si menciona "importante", priority=2
- Por defecto, priority=1
"""

        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Extraés tareas de texto natural. Respondé SOLO JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )

            content = r.choices[0].message.content.strip()
            content = content.replace("```json", "").replace("```", "").strip()
            result = json.loads(content)

            return result

        except Exception as e:
            print(f"Error en extract_task_smart: {e}")
            from .utils import parse_task_nl
            return parse_task_nl(text)

    # ---------- ANÁLISIS DE SENTIMIENTOS CON CONTEXTO -------------

    def analyze_sentiment_contextual(self, text: str, user_id: str) -> Dict:
        """
        Análisis emocional que considera historial y patrones
        """
        db = self._db()
        user = ensure_user(db, user_id)
        recent_moods = user.get("moods", [])[-5:]

        mood_context = ""
        if recent_moods:
            avg_recent = sum(m["score"]
                             for m in recent_moods) / len(recent_moods)
            mood_context = f"Estado emocional reciente: {sentiment_bucket(avg_recent)}"

        prompt = f"""Analiza el estado emocional en este mensaje:

{mood_context}

Mensaje: "{text}"

Devolve JSON:
{{
  "score": <-1.0 a 1.0>,
  "label": "<positivo|neutral|negativo>",
  "intensity": "<bajo|medio|alto>",
  "needs_support": <true si parece necesitar apoyo emocional>,
  "suggested_response_tone": "<empático|motivador|celebratorio|neutral>"
}}
"""

        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Analizas emociones. Respondé SOLO JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )

            content = r.choices[0].message.content.strip()
            content = content.replace("```json", "").replace("```", "").strip()
            result = json.loads(content)

            user["moods"].append(
                {
                    "ts": datetime.now().isoformat(),
                    "score": result.get("score", 0),
                    "text": text[:200],
                }
            )
            self._save(db)

            return result

        except Exception as e:
            print(f"Error en analyze_sentiment_contextual: {e}")
            return {
                "score": 0.0,
                "label": "neutral",
                "intensity": "bajo",
                "needs_support": False,
                "suggested_response_tone": "neutral",
            }

    # ---------- RESPUESTA INTELIGENTE CON CONTEXTO -------------

    def generate_smart_response(
        self, text: str, intent: Dict, sentiment: Dict, user_id: str
    ) -> str:
        """
        Genera respuestas más naturales y contextuales
        """
        db = self._db()
        user = ensure_user(db, user_id)

        pending = [t for t in user["tasks"] if not t["done"]]
        completed_today = [
            t
            for t in user["tasks"]
            if t.get("done")
            and t.get("completed_at", "")[:10] == datetime.now().date().isoformat()
        ]

        context = f"""Contexto del usuario:
- Tareas pendientes: {len(pending)}
- Completadas hoy: {len(completed_today)}
- Estado emocional: {sentiment['label']} (intensidad: {sentiment['intensity']})
- Tono sugerido: {sentiment['suggested_response_tone']}

Intención detectada: {intent['intent']}

Mensaje del usuario: "{text}"
"""

        system_prompt = """Sos un asistente personal cálido y humano. Tu trabajo es:
- Responder de forma natural, sin ser robótico
- Ser breve pero significativo
- Adaptar tu tono al estado emocional del usuario
- Ofrecer ayuda concreta cuando corresponda
- NO usar emojis en exceso (máximo 1-2)
- Usar lenguaje argentino informal pero respetuoso

Si el usuario está mal, ofrecé apoyo real.
Si logró algo, celebra genuinamente.
Si está perdido, guialo sin regañarlo."""

        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context},
                ],
                temperature=0.8,
                max_tokens=150,
            )

            return r.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error en generate_smart_response: {e}")
            return "Perdón, tuve un problema. ¿Probamos de nuevo?"

    # ---------- DETECCIÓN Y EXTRACCIÓN DE MÚLTIPLES TAREAS -------------

    def detect_multiple_tasks(self, text: str) -> bool:
        """Detecta si el usuario está intentando crear múltiples tareas"""
        patterns = [
            r"\d+[\.\-\)]\s*\w+",
            r"•\s*\w+",
            r"-\s*\w+.*\n.*-\s*\w+",
        ]

        for pattern in patterns:
            if re.search(pattern, text):
                return True

        if text.count(" y ") >= 2 or text.count(",") >= 2:
            return True

        return False

    def extract_multiple_tasks(self, text: str) -> List[Dict]:
        """Extrae múltiples tareas de un texto con lista"""
        prompt = f"""El usuario quiere crear VARIAS tareas a la vez. Extraé cada una por separado.

Texto: "{text}"

Devolve SOLO un JSON array:
[
  {{
    "title": "<título de la tarea 1>",
    "priority": <1-3>
  }},
  {{
    "title": "<título de la tarea 2>",
    "priority": <1-3>
  }}
]

Reglas:
- Cada tarea debe ser clara y accionable
- NO incluyas números de lista (1., 2., etc)
- Limpia y mejora los títulos
- Las tareas son agendadas para HOY (no tienen fecha futura)
"""

        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Extraés múltiples tareas de texto. Respondé SOLO JSON array.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )

            content = r.choices[0].message.content.strip()
            content = content.replace("```json", "").replace("```", "").strip()
            tasks_data = json.loads(content)

            if not isinstance(tasks_data, list):
                return []

            return tasks_data

        except Exception as e:
            print(f"Error en extract_multiple_tasks: {e}")
            return []

    def extract_multiple_reminders(self, text: str) -> List[Dict]:
        """Extrae múltiples recordatorios de un texto"""
        prompt = f"""El usuario quiere crear VARIOS recordatorios. Extraé cada uno.

Texto: "{text}"

Devolve SOLO un JSON array:
[
  {{
    "title": "<qué recordar>",
    "time_expression": "<cuándo recordar>"
  }},
  {{
    "title": "<qué recordar>",
    "time_expression": "<cuándo recordar>"
  }}
]

Reglas:
- NO incluyas "recordame" en los títulos
- Mantené las expresiones temporales exactas
"""

        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Extraés múltiples recordatorios. Respondé SOLO JSON array.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )

            content = r.choices[0].message.content.strip()
            content = content.replace("```json", "").replace("```", "").strip()
            reminders_data = json.loads(content)

            if not isinstance(reminders_data, list):
                return []

            return reminders_data

        except Exception as e:
            print(f"Error en extract_multiple_reminders: {e}")
            return []

    # ---------- MÉTODOS DE CREACIÓN DE TAREAS -------------

    def add_multiple_tasks(self, user_id: str, text: str) -> str:
        """Crea múltiples tareas a la vez"""
        db = self._db()
        user = ensure_user(db, user_id)

        tasks_data = self.extract_multiple_tasks(text)

        if not tasks_data:
            return self.add_task_smart(user_id, text, {})

        now = datetime.now().isoformat()
        added = []

        for task_data in tasks_data:
            task = {
                "title": task_data.get("title", "Tarea"),
                "priority": task_data.get("priority", 1),
                "notes": "",
                "created_at": now,
                "done": False,
            }
            user["tasks"].append(task)
            added.append(task["title"])

        user["last_added_task"] = len(user["tasks"]) - 1

        user["history"].append(
            {
                "ts": now,
                "type": "multiple_tasks_add",
                "raw": text,
                "count": len(added),
            }
        )

        self._save(db)

        if len(added) == 1:
            return f"✅ Listo, agendé para hoy: *{added[0]}*"
        else:
            lista = "\n".join(f"  {i+1}. {title}" for i,
                              title in enumerate(added))
            return f"✅ Perfecto, agendé {len(added)} tareas para hoy:\n\n{lista}"

    def add_task_smart(self, user_id: str, text: str, intent_data: Dict) -> str:
        """Versión mejorada que usa extracción inteligente"""
        if self.detect_multiple_tasks(text):
            return self.add_multiple_tasks(user_id, text)

        db = self._db()
        user = ensure_user(db, user_id)

        task_data = self.extract_task_smart(text, intent_data)

        task = {
            "title": task_data.get("title", "Tarea"),
            "priority": task_data.get("priority", 1),
            "notes": task_data.get("notes", ""),
            "created_at": datetime.now().isoformat(),
            "done": False,
        }

        user["tasks"].append(task)
        user["last_added_task"] = len(user["tasks"]) - 1

        user["history"].append(
            {
                "ts": datetime.now().isoformat(),
                "type": "task_add",
                "raw": text,
                "parsed": task,
            }
        )

        self._save(db)

        return f"Listo, anoté para hoy: *{task['title']}*"

    # ---------- LISTAR TAREAS -------------

    def list_tasks_smart(self, user_id: str, scope: str = "pending") -> str:
        """
        Lista tareas según scope:
        - pending: solo pendientes (por defecto)
        - all: todas las tareas
        - completed: solo completadas
        """
        db = self._db()
        user = ensure_user(db, user_id)

        all_tasks = user.get("tasks", [])

        if scope == "all":
            tasks = all_tasks
            header = "Todas tus tareas:"
        elif scope == "completed":
            tasks = [t for t in all_tasks if t.get("done")]
            header = "Tareas completadas:"
        else:  # pending
            tasks = [t for t in all_tasks if not t.get("done")]
            header = "Estas son tus tareas de hoy:"

        if not tasks:
            if scope == "completed":
                return "No tenés tareas completadas todavía."
            elif scope == "all":
                return "No tenés tareas cargadas todavía."
            return "No tenés tareas pendientes hoy 🎉"

        out = [header]

        for i, t in enumerate(tasks, start=1):
            prefix = "⚠️ " if t.get("priority", 1) >= 3 else ""
            status = "✅ " if t.get("done") else ""
            out.append(f"{i}. {status}{prefix}{t['title']}")

        return "\n".join(out)

    def suggest_task_order(self, user_id: str) -> str:
        """
        Sugiere un orden de ejecución para las tareas pendientes,
        basado en prioridad (3 > 2 > 1).
        """
        db = self._db()
        user = ensure_user(db, user_id)

        pending = [t for t in user.get("tasks", []) if not t.get("done")]

        if not pending:
            return "No tenés tareas pendientes hoy como para sugerir un orden 🙂"

        def sort_key(t):
            return -t.get("priority", 1)

        ordered = sorted(pending, key=sort_key)

        out = ["Te sugiero este orden para hoy:"]
        for i, t in enumerate(ordered, start=1):
            urgency = ""
            if t.get("priority", 1) == 3:
                urgency = " ⚠️ URGENTE"
            elif t.get("priority", 1) == 2:
                urgency = " 🔸 Importante"

            out.append(f"{i}. {t['title']}{urgency}")

        out.append("\nTomalo como guía, podés cambiarlo si te sirve más 😉")
        return "\n".join(out)

    # ---------- MARCAR TAREAS COMO COMPLETADAS -------------

    def mark_done(self, user_id: str, idx: int) -> str:
        """Marca UNA tarea como completada por índice"""
        db = self._db()
        user = ensure_user(db, user_id)

        pending_indices = []
        for i, t in enumerate(user["tasks"]):
            if not t.get("done", False):
                pending_indices.append(i)

        if idx < 1 or idx > len(pending_indices):
            return "Ese número no existe. Usá /tasks para ver la lista."

        real_index = pending_indices[idx - 1]
        task = user["tasks"][real_index]

        task["done"] = True
        task["completed_at"] = datetime.now().isoformat()

        user["history"].append(
            {"ts": datetime.now().isoformat(), "type": "task_done",
             "task": task["title"]}
        )

        self._save(db)

        db_fresh = self._db()
        user_fresh = db_fresh.get("users", {}).get(user_id, {})
        completed_today = len(
            [
                t
                for t in user_fresh.get("tasks", [])
                if t.get("completed_at", "")[:10] == datetime.now().date().isoformat()
            ]
        )

        msg = f"💪 ¡Genial! Tachaste: *{task['title']}*"
        if completed_today >= 3:
            msg += f"\n\nYa llevas {completed_today} tareas hoy. ¡Imparable!"

        return msg

    def mark_all_done(self, user_id: str) -> str:
        """Marca TODAS las tareas pendientes como completadas"""
        db = self._db()
        user = ensure_user(db, user_id)

        pending = [t for t in user["tasks"] if not t["done"]]

        if not pending:
            return "No tenés tareas pendientes para marcar 🤔"

        count = len(pending)
        now = datetime.now().isoformat()

        for task in pending:
            task["done"] = True
            task["completed_at"] = now

        user["history"].append(
            {"ts": now, "type": "mark_all_done", "count": count}
        )

        self._save(db)

        if count == 1:
            return f"✅ Perfecto, marqué *{pending[0]['title']}* como completada."
        else:
            return (
                f"🎉 ¡Increíble! Marqué todas tus {count} tareas como completadas.\n\n"
                "¿Te tomás un descanso o seguimos?"
            )

    def mark_multiple_done(self, user_id: str, indices: list) -> str:
        """Marca varias tareas específicas como completadas"""
        db = self._db()
        user = ensure_user(db, user_id)

        pending = [t for t in user["tasks"] if not t["done"]]

        if not pending:
            return "No tenés tareas pendientes."

        completed = []
        invalid = []

        for idx in indices:
            if idx < 1 or idx > len(pending):
                invalid.append(idx)
            else:
                task = pending[idx - 1]
                if task not in completed:
                    task["done"] = True
                    task["completed_at"] = datetime.now().isoformat()
                    completed.append(task)

        self._save(db)

        msgs = []
        if completed:
            if len(completed) == 1:
                msgs.append(f"✅ Marqué: *{completed[0]['title']}*")
            else:
                msgs.append("✅ Marqué varias tareas:")
                for t in completed:
                    msgs.append(f"  • {t['title']}")

        if invalid:
            msgs.append(
                f"\n⚠️ Números inválidos: {', '.join(map(str, invalid))}")

        return "\n".join(msgs)

    # ---------- RECORDATORIOS (REMINDERS) ----------

    def add_reminder_smart(self, user_id: str, text: str) -> str:
        """
        Crea un recordatorio usando extracción inteligente.
        """
        db = self._db()
        user = ensure_user(db, user_id)

        reminder_data = self.extract_reminder_smart(text)

        time_expr = reminder_data.get("time_expression")
        if not time_expr:
            return "No pude entender cuándo querés que te recuerde. ¿Me lo decís de nuevo?"

        remind_dt = parse_datetime_in_text(time_expr)

        if not remind_dt:
            return "No pude entender el tiempo. Probá con 'en 5 minutos', 'a las 15:30', etc."

        now = datetime.now()
        if remind_dt <= now:
            return f"Esa hora ya pasó ({friendly_due(remind_dt.isoformat())}). ¿Querés que sea para más adelante?"

        reminder = {
            "title": reminder_data.get("title", "Recordatorio"),
            "remind_datetime": remind_dt.isoformat(),
            "created_at": now.isoformat(),
            "reminded": False,
        }

        user["reminders"].append(reminder)

        user["history"].append(
            {
                "ts": now.isoformat(),
                "type": "reminder_add",
                "raw": text,
                "parsed": reminder,
            }
        )

        self._save(db)

        return f"Perfecto, agendé: *{reminder['title']}* para el {friendly_due(remind_dt.isoformat())} ✓"

    def add_multiple_reminders(self, user_id: str, text: str) -> str:
        """Crea múltiples recordatorios a la vez"""
        db = self._db()
        user = ensure_user(db, user_id)

        reminders_data = self.extract_multiple_reminders(text)

        if not reminders_data:
            return self.add_reminder_smart(user_id, text)

        now = datetime.now()
        added = []
        failed = []

        for reminder_data in reminders_data:
            time_expr = reminder_data.get("time_expression")
            if not time_expr:
                failed.append(reminder_data.get("title", "?"))
                continue

            remind_dt = parse_datetime_in_text(time_expr)
            if not remind_dt or remind_dt <= now:
                failed.append(reminder_data.get("title", "?"))
                continue

            reminder = {
                "title": reminder_data.get("title", "Recordatorio"),
                "remind_datetime": remind_dt.isoformat(),
                "created_at": now.isoformat(),
                "reminded": False,
            }

            user["reminders"].append(reminder)
            added.append(
                f"{reminder['title']} ({friendly_due(remind_dt.isoformat())})")

        user["history"].append(
            {
                "ts": now.isoformat(),
                "type": "multiple_reminders_add",
                "raw": text,
                "count": len(added),
            }
        )

        self._save(db)

        if not added:
            return "No pude crear ningún recordatorio. Revisá las fechas/horas."

        lista = "\n".join(f"  {i+1}. {r}" for i, r in enumerate(added))

        msg = f"✅ Perfecto, agendé {len(added)} recordatorios:\n\n{lista}"

        if failed:
            msg += f"\n\n⚠️ No pude agendar: {', '.join(failed)}"

        return msg

    def list_reminders(self, user_id: str) -> str:
        """Lista todos los recordatorios programados."""
        db = self._db()
        user = ensure_user(db, user_id)

        reminders = user.get("reminders", [])

        if not reminders:
            return "No tenés recordatorios programados."

        out = ["📅 *Recordatorios programados:*"]
        for i, r in enumerate(reminders, start=1):
            remind_iso = r.get("remind_datetime")
            remind_str = friendly_due(
                remind_iso) if remind_iso else "sin fecha"
            out.append(f"{i}. *{r['title']}*")
            out.append(f"   ⏰ Te voy a avisar: {remind_str}")

        return "\n".join(out)

    def delete_reminder(self, user_id: str, index: int) -> str:
        """Elimina un recordatorio por número."""
        db = self._db()
        user = ensure_user(db, user_id)

        reminders = user.get("reminders", [])

        if index < 1 or index > len(reminders):
            return "Ese número no existe. Usá /reminders para ver la lista."

        removed = reminders.pop(index - 1)
        self._save(db)

        return f"Eliminé el recordatorio: *{removed['title']}*"

    def delete_all_reminders(self, user_id: str) -> str:
        """Elimina TODOS los recordatorios."""
        db = self._db()
        user = ensure_user(db, user_id)

        count = len(user.get("reminders", []))
        user["reminders"] = []
        self._save(db)

        return f"Listo, eliminé {count} recordatorio(s)."

    def delete_reminder_by_text(self, user_id: str, text: str) -> str:
        """
        Borra un recordatorio buscando por texto (ej: 'turno con la oculista').
        """
        db = self._db()
        user = ensure_user(db, user_id)
        reminders = user.get("reminders", [])

        if not reminders:
            return "No tenés recordatorios programados."

        lower = text.lower()

        # Palabras "relevantes" del mensaje
        words = [w for w in re.split(r"[^\wáéíóúñ]+", lower) if len(w) >= 4]

        def matches(r):
            title = r.get("title", "").lower()
            return any(w in title for w in words)

        matched_indices = [i for i, r in enumerate(reminders) if matches(r)]

        if len(matched_indices) == 0:
            return (
                "No encontré ningún recordatorio que coincida con eso.\n"
                "Usá /reminders para ver la lista y /delete_reminder N para borrar uno puntual."
            )

        if len(matched_indices) > 1:
            lista = "\n".join(
                f"• {reminders[i]['title']}" for i in matched_indices)
            return (
                "Tengo más de un recordatorio que puede coincidir:\n"
                f"{lista}\n\n"
                "Usá /reminders para ver los números y /delete_reminder N para borrar el que quieras."
            )

        idx = matched_indices[0]
        removed = reminders.pop(idx)
        self._save(db)

        return f"Eliminé el recordatorio: *{removed['title']}*"

    def reschedule_reminder_by_text(self, user_id: str, text: str) -> str:
        """
        Cambia la fecha/hora de un recordatorio según el mensaje.
        """
        db = self._db()
        user = ensure_user(db, user_id)
        reminders = user.get("reminders", [])

        if not reminders:
            return "No tenés recordatorios programados."

        new_dt = parse_datetime_in_text(text)
        if not new_dt:
            return (
                "No me quedó clara la nueva fecha/hora del recordatorio.\n"
                "Probá con algo como \"para el martes a las 9\" o \"para mañana a las 18\"."
            )

        if new_dt <= datetime.now():
            return "La nueva hora que me diste ya pasó. Probá con un horario a futuro 🙂"

        lower = text.lower()
        words = [w for w in re.split(r"[^\wáéíóúñ]+", lower) if len(w) >= 4]

        def matches(r):
            title = r.get("title", "").lower()
            return any(w in title for w in words)

        matched_indices = [i for i, r in enumerate(reminders) if matches(r)]

        if len(matched_indices) == 0:
            idx = len(reminders) - 1
        elif len(matched_indices) == 1:
            idx = matched_indices[0]
        else:
            lista = "\n".join(
                f"• {reminders[i]['title']}" for i in matched_indices)
            return (
                "Tengo más de un recordatorio que puede coincidir con eso:\n"
                f"{lista}\n\n"
                "Usá /reminders para ver la lista y decime, por ejemplo,\n"
                "\"moví el recordatorio 2 para mañana a las 9\"."
            )

        r = reminders[idx]
        r["remind_datetime"] = new_dt.isoformat()
        self._save(db)

        return f"Listo, moví el recordatorio de *{r['title']}* a {friendly_due(r['remind_datetime'])}."

    def get_due_reminders(self):
        """
        Devuelve recordatorios cuyo remind_datetime YA ocurrió
        y los elimina de la lista de recordatorios programados.

        Formato devuelto: { user_id: [reminder1, reminder2...] }
        """
        db = self._db()
        now = datetime.now()

        due = {}

        for uid, user in db.get("users", {}).items():
            remaining = []

            for r in user.get("reminders", []):
                remind_iso = r.get("remind_datetime")
                if not remind_iso:
                    remaining.append(r)
                    continue

                try:
                    dt = datetime.fromisoformat(remind_iso)
                except Exception:
                    remaining.append(r)
                    continue

                if dt <= now:
                    due.setdefault(uid, []).append(r)
                else:
                    remaining.append(r)

            user["reminders"] = remaining

        self._save(db)
        return due

    # ---------- ESTADÍSTICAS Y RESUMEN -------------

    def get_stats(self, user_id: str) -> str:
        """Obtiene estadísticas de productividad del usuario"""
        db = self._db()
        user = ensure_user(db, user_id)

        today = datetime.now().date().isoformat()

        all_tasks = user.get("tasks", [])
        total = len(all_tasks)
        pending = len([t for t in all_tasks if not t.get("done")])
        completed = total - pending

        completed_today = [
            t for t in all_tasks
            if t.get("completed_at", "")[:10] == today
        ]

        urgent_pending = [
            t for t in all_tasks
            if not t.get("done") and t.get("priority", 1) >= 3
        ]

        # Calcular racha de días consecutivos completando tareas
        dates_with_completions = set()
        for t in all_tasks:
            if t.get("completed_at"):
                dates_with_completions.add(t.get("completed_at")[:10])

        out = [
            "📊 *Tus estadísticas:*\n",
            f"🎯 Total de tareas: {total}",
            f"✅ Completadas: {completed}",
            f"⏳ Pendientes: {pending}",
            f"",
            f"*Hoy ({datetime.now().strftime('%d/%m')}):*",
            f"✓ Completaste: {len(completed_today)} tareas",
        ]

        if urgent_pending:
            out.append(f"⚠️ Urgentes pendientes: {len(urgent_pending)}")

        if completed > 0:
            completion_rate = (completed / total) * 100
            out.append(f"\n💪 Tasa de finalización: {completion_rate:.1f}%")

        if len(dates_with_completions) > 0:
            out.append(f"🔥 Días con actividad: {len(dates_with_completions)}")

        return "\n".join(out)

    def reflect_today(self, user_id: str) -> str:
        """Resumen mejorado del día"""
        db = self._db()
        user = ensure_user(db, user_id)

        today = datetime.now().date().isoformat()
        moods = [m for m in user["moods"] if m["ts"][:10] == today]
        done = [
            t
            for t in user["tasks"]
            if t.get("completed_at", "")[:10] == today
        ]
        pending = [t for t in user["tasks"] if not t["done"]]

        if moods:
            avg = sum(m["score"] for m in moods) / len(moods)
            mood_txt = sentiment_bucket(avg)
        else:
            mood_txt = "sin datos"

        out = [
            "📊 *Tu día hasta ahora:*",
            f"Estado: {mood_txt}",
            f"Completaste: {len(done)} {'tarea' if len(done) == 1 else 'tareas'}",
            f"Te quedan: {len(pending)}",
        ]

        if pending:
            out.append("\n¿Arrancamos con alguna?")
        else:
            out.append("\n¡Día despejado! 🎉")

        return "\n".join(out)

    # ---------- COMPATIBILIDAD CON CÓDIGO VIEJO -------------

    def coaching_reply(self, text: str, mood: str) -> str:
        """Backward compatibility"""
        intent = {"intent": "chat", "extracted_data": {}}
        sentiment = {
            "label": mood,
            "intensity": "medio",
            "suggested_response_tone": "neutral",
        }
        return self.generate_smart_response(text, intent, sentiment, "default")
