import os
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
import dateparser

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from app.chat import ChatManager

chat = ChatManager()

# ------------------------------
# FECHA ÚLTIMO MENSAJE (para no saludar siempre)
# ------------------------------
last_message_time = {}

def should_greet(uid: str) -> bool:
    now = datetime.now()
    last = last_message_time.get(uid)

    if not last:
        last_message_time[uid] = now
        return True

    diff = (now - last).total_seconds()
    last_message_time[uid] = now

    return diff > 300  # 5 minutos


# ------------------------------
# EXTRACCIÓN DE FECHAS PARA CONSULTAS DE AGENDA
# ------------------------------
def extract_date_question(text: str) -> datetime | None:
    text = text.lower()

    if "hoy" in text:
        return datetime.now()

    if "mañana" in text or "manana" in text:
        return datetime.now() + timedelta(days=1)

    if "pasado mañana" in text or "pasado manana" in text:
        return datetime.now() + timedelta(days=2)

    dt = dateparser.parse(text, languages=["es"])
    return dt


def tasks_for_date(user_id: str, target_date: datetime):
    db = chat._db()
    user = db.get("users", {}).get(user_id, {})
    tasks = user.get("tasks", [])

    if not tasks:
        return "No tenés tareas registradas."

    out = []
    for t in tasks:
        due = t.get("due")
        if not due:
            continue
        try:
            dt = datetime.fromisoformat(due)
            if dt.date() == target_date.date():
                hora = dt.strftime("%H:%M")
                out.append(f"• {t['title']} — {hora}")
        except:
            pass

    if not out:
        return f"No tenés tareas para {target_date.strftime('%d/%m')}."

    return (
        f"Tareas para *{target_date.strftime('%d/%m')}*:\n" +
        "\n".join(out)
    )


# ------------------------------
# COMANDOS
# ------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = str(update.effective_user.id)

    if should_greet(uid):
        await update.message.reply_text(
            "¡Hola! Soy tu organizador ✨\n"
            "Contame qué te queda por hacer o cómo te sentís."
        )
    else:
        await update.message.reply_text("¿En qué te doy una mano? 😊")


async def tasks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = str(update.effective_user.id)
    await update.message.reply_text(chat.list_tasks(uid), parse_mode="Markdown")


async def today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = str(update.effective_user.id)
    await update.message.reply_text(chat.reflect_today(uid), parse_mode="Markdown")


async def done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = str(update.effective_user.id)

    if not context.args:
        await update.message.reply_text("Decime /done N")
        return

    try:
        idx = int(context.args[0])
    except:
        await update.message.reply_text("Pasame un número válido")
        return

    await update.message.reply_text(chat.mark_done(uid, idx), parse_mode="Markdown")


async def notify_due_tasks(context: ContextTypes.DEFAULT_TYPE):
    due = chat.get_due_tasks_for_reminder()

    for uid, tasks in due.items():
        lista = "\n".join(f"• {t['title']}" for t in tasks)
        text = (
            "⏰ *Recordatorio de hoy:*\n"
            f"{lista}\n\n"
            "¿Querés marcar algo como hecho?"
        )
        await context.bot.send_message(chat_id=int(uid), text=text, parse_mode="Markdown")


# ------------------------------
# MANEJO DEL TEXTO
# ------------------------------
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = str(update.effective_user.id)
    raw = update.message.text.strip()
    text = raw.lower().strip()

    last_message_time[uid] = datetime.now()

    # CONFIRMACIÓN DE RECORDATORIO
    yes_words = ["si", "sí", "sii", "ya", "ya esta", "ya hice", "ya la hice", "ya lo hice"]
    if text in yes_words or text.startswith("ya "):
        resp = chat.mark_last_reminded_done(uid)
        if resp:
            await update.message.reply_text(resp, parse_mode="Markdown")
            return

    # PREGUNTAS DE AGENDA
    if any(w in text for w in ["qué tengo", "que tengo", "qué hay", "que hay", "qué tareas", "que tareas"]):
        date_q = extract_date_question(text)
        if date_q:
            resp = tasks_for_date(uid, date_q)
            await update.message.reply_text(resp, parse_mode="Markdown")
            return
        else:
            await update.message.reply_text("Decime una fecha o día y te digo 🙂")
            return

    # CREAR NUEVA TAREA
    if any(w in text for w in ["recordame", "avisame", "tengo que", "debo", "necesito que"]):
        resp = chat.add_task_from_text(uid, raw)
        await update.message.reply_text(resp, parse_mode="Markdown")
        return

    # MODIFICAR FECHA DE LA ÚLTIMA TAREA
    if any(w in text for w in ["agendalo", "agéndalo", "ponelo", "movelo", "pasalo", "cambialo"]):
        resp = chat.edit_last_task_due(uid, raw)
        await update.message.reply_text(resp, parse_mode="Markdown")
        return

    # CHAT / COACHING (ahora mucho más humano)
    score, mood = chat.analyze_sentiment(raw)
    reply = chat.coaching_reply(raw, mood)
    await update.message.reply_text(reply, parse_mode="Markdown")


# ------------------------------
# MAIN
# ------------------------------
def main():
    load_dotenv()
    token = os.getenv("TELEGRAM_TOKEN")

    logging.basicConfig(level=logging.INFO)

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("tasks", tasks))
    app.add_handler(CommandHandler("today", today))
    app.add_handler(CommandHandler("done", done))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    app.job_queue.run_repeating(notify_due_tasks, interval=60, first=10)

    print("Bot corriendo 💛")
    app.run_polling()


if __name__ == "__main__":
    main()
