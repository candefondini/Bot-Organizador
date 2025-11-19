import os
import logging
import asyncio
from datetime import datetime, timedelta
from app.utils import friendly_due, parse_datetime_in_text
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
#Este archivo conecta nuestro bot con Telegram: define los comandos,
#maneja lo que escribe el usuario y delega toda la parte inteligente 
# (entender intenciones, emociones, tareas y recordatorios) en la clase ChatManager
from app.chat import ChatManager

chat = ChatManager()

# ------------------------------
# GESTIÓN DE CONTEXTO
# ------------------------------
last_message_time = {}
conversation_state = {}
processing_messages = {}


def should_greet(uid: str) -> bool:
    now = datetime.now()
    last = last_message_time.get(uid)

    if not last:
        last_message_time[uid] = now
        return True

    diff = (now - last).total_seconds()
    last_message_time[uid] = now

    return diff > 1800  # 30 minutos


# ------------------------------
# COMANDOS
# ------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = str(update.effective_user.id)

    await update.message.reply_text(
        "¡Buenas! Soy tu asistente personal. ¡Encantado de conocerte!\n\n"
        "Puedo ayudarte a:\n"
        "📋 Organizar tus tareas del día\n"
        "⏰ Crear recordatorios con avisos\n"
        "📊 Mostrarte estadísticas de productividad\n"
        "💡 Sugerir orden de prioridades\n\n"
        "Si necesitás ayuda simplemente escribí /help y te muestro los comandos disponibles.\n\n"
        "Hablame natural, yo entiendo 😊"
    )


async def tasks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = str(update.effective_user.id)
    
    # Si pasan argumentos, manejamos diferentes scopes
    scope = "pending"
    if context.args:
        arg = context.args[0].lower()
        if arg in ["todas", "all", "todo"]:
            scope = "all"
        elif arg in ["completadas", "hechas", "terminadas"]:
            scope = "completed"
    
    await update.message.reply_text(
        chat.list_tasks_smart(uid, scope),
    )


async def today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = str(update.effective_user.id)
    await update.message.reply_text(
        chat.reflect_today(uid),
        parse_mode="Markdown",
    )


async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Muestra estadísticas de productividad"""
    uid = str(update.effective_user.id)
    await update.message.reply_text(
        chat.get_stats(uid),
        parse_mode="Markdown",
    )


async def done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = str(update.effective_user.id)

    if not context.args:
        tasks_list = chat.list_tasks_smart(uid)
        await update.message.reply_text(
            f"{tasks_list}\n\n"
            "Decime qué terminaste:\n"
            "• /done 1 - marca la tarea 1\n"
            "• /done 1 2 3 - marca varias\n"
            "• /done all - marca todas",
        )
        return

    # Caso especial: /done all
    if context.args[0].lower() in ["all", "todas", "todo"]:
        response = chat.mark_all_done(uid)
        await update.message.reply_text(response, parse_mode="Markdown")
        return

    indices = []
    invalid = []

    for arg in context.args:
        try:
            idx = int(arg)
            indices.append(idx)
        except Exception:
            invalid.append(arg)

    if invalid:
        await update.message.reply_text(
            f"No entendí: {', '.join(invalid)}\n"
            "Pasame números o 'all' para marcar todas."
        )
        return

    if not indices:
        await update.message.reply_text("Pasame al menos un número válido 🙂")
        return

    if len(indices) == 1:
        response = chat.mark_done(uid, indices[0])
    else:
        response = chat.mark_multiple_done(uid, indices)

    await update.message.reply_text(response, parse_mode="Markdown")


async def notify_due_reminders(context: ContextTypes.DEFAULT_TYPE):
    """Enviar mensajes cuando toca un recordatorio."""
    due = chat.get_due_reminders()

    for uid, reminders in due.items():
        for r in reminders:
            text = f"⏰ *Recordatorio:* {r['title']}"
            
            await context.bot.send_message(
                chat_id=int(uid),
                text=text,
                parse_mode="Markdown",
            )


# ------------------------------
# MANEJO INTELIGENTE DEL TEXTO
# ------------------------------
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = str(update.effective_user.id)
    raw = update.message.text.strip()

    # Detectar múltiples tareas/recordatorios en formato lista
    if any(st in raw for st in ["1.", "1)", "•", "-", "*", "\n2", "\n3"]):
        # Determinar si son tareas o recordatorios según el contexto
        lower = raw.lower()
        if any(w in lower for w in ["recordame", "recordar", "avisame", "avisar"]):
            response = chat.add_multiple_reminders(uid, raw)
        else:
            response = chat.add_multiple_tasks(uid, raw)
        
        await update.message.reply_text(response, parse_mode="Markdown")
        return

    message_id = update.message.message_id
    message_key = f"{uid}_{message_id}"
    
    if message_key in processing_messages:
        logging.warning(f"Mensaje {message_id} duplicado, ignorando")
        return

    processing_messages[message_key] = True

    try:
        last_message_time[uid] = datetime.now()

        conv_context = chat._get_conversation_context(uid)

        intent = chat.classify_intent(raw, conv_context)

        sentiment = chat.analyze_sentiment_contextual(raw, uid)

        confidence = intent.get("confidence", 0)
        intent_type = intent.get("intent", "chat")

        if confidence < 0.6 and intent_type in ["create_task", "create_reminder"]:
            response = chat.generate_smart_response(raw, intent, sentiment, uid)
            await update.message.reply_text(response, parse_mode="Markdown")
            return

        # CREAR RECORDATORIO (modo natural)
        if intent_type == "create_reminder":
            response = chat.add_reminder_smart(uid, raw)
            await update.message.reply_text(response, parse_mode="Markdown")
            return

        # CREAR TAREA (modo natural)
        if intent_type == "create_task":
            response = chat.add_task_smart(uid, raw, intent.get("extracted_data", {}))
            await update.message.reply_text(response, parse_mode="Markdown")
            return

        # CONSULTAR TAREAS
        if intent_type == "query_tasks":
            scope = intent.get("extracted_data", {}).get("query_scope", "pending")
            tasks_list = chat.list_tasks_smart(uid, scope)
            await update.message.reply_text(tasks_list)
            return

        # CONSULTAR RECORDATORIOS
        if intent_type == "query_reminders":
            reminders_text = chat.list_reminders(uid)
            await update.message.reply_text(reminders_text, parse_mode="Markdown")
            return

        # CONSULTAR ESTADÍSTICAS
        if intent_type == "query_stats":
            stats_text = chat.get_stats(uid)
            await update.message.reply_text(stats_text, parse_mode="Markdown")
            return

        # MARCAR COMO HECHA
        if intent_type == "mark_done":
            await update.message.reply_text(
                "¿Cuál terminaste? Usá /tasks para ver la lista y después /done N",
            )
            return

        # MARCAR TODAS COMO HECHAS
        if intent_type == "mark_all_done":
            response = chat.mark_all_done(uid)
            await update.message.reply_text(response, parse_mode="Markdown")
            return

        # ELIMINAR RECORDATORIO
        if intent_type == "delete_reminder":
            response = chat.delete_reminder_by_text(uid, raw)
            await update.message.reply_text(response, parse_mode="Markdown")
            return

        # MODIFICAR RECORDATORIO
        if intent_type == "modify_reminder":
            response = chat.reschedule_reminder_by_text(uid, raw)
            await update.message.reply_text(response, parse_mode="Markdown")
            return

        # EXPRESAR EMOCIÓN O CHAT
        if intent_type in ["express_emotion", "chat"]:
            response = chat.generate_smart_response(raw, intent, sentiment, uid)
            await update.message.reply_text(response)

            if sentiment.get("needs_support"):
                db = chat._db()
                user = db.get("users", {}).get(uid, {})
                pending = [t for t in user.get("tasks", []) if not t["done"]]

                if len(pending) > 5:
                    await update.message.reply_text(
                        "Por si sirve, veo que tenés varias cosas pendientes. "
                        "¿Te ayudo a priorizarlas?"
                    )
            return

        # FALLBACK
        response = chat.generate_smart_response(raw, intent, sentiment, uid)
        await update.message.reply_text(response)

    finally:
        await asyncio.sleep(5)
        processing_messages.pop(message_key, None)


# ------------------------------
# COMANDOS ADICIONALES
# ------------------------------
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "📚 *Comandos disponibles:*\n\n"
        "*Tareas del día:*\n"
        "/tasks - Ver tus tareas pendientes\n"
        "/tasks todas - Ver todas las tareas\n"
        "/tasks completadas - Ver solo completadas\n"
        "/add tarea - Añadir una tarea nueva\n"
        "/done N - Marcar tarea N como hecha\n"
        "/done all - Marcar todas como hechas\n"
        "/suggestion - Te sugiero orden de prioridad\n\n"
        "*Recordatorios:*\n"
        "/reminders - Ver recordatorios programados\n"
        "/delete\\_reminder N - Eliminar recordatorio N\n"
        "/delete\\_all\\_reminders - Eliminar todos\n\n"
        "*Estadísticas:*\n"
        "/today - Resumen de tu día\n"
        "/stats - Estadísticas completas\n\n"
        "/help - Este mensaje\n\n"
        "━━━━━━━━━━━━━━━━━━━━━\n\n"
        "*También podés hablarme natural:*\n\n"
        "*Para tareas:*\n"
        "• \"hoy tengo que comprar pan\"\n"
        "• \"limpiar mi pieza\"\n"
        "• \"hoy: 1\\- X 2\\- Y 3\\- Z\" (varias)\n\n"
        "*Para recordatorios:*\n"
        "• \"recordame llamar a Juan en 5 minutos\"\n"
        "• \"avisame a las 15hs reunión\"\n"
        "• \"borrá el recordatorio del turno\"\n"
        "• \"mové el recordatorio para mañana\"\n\n"
        "*Consultas:*\n"
        "• \"qué tengo para hoy?\"\n"
        "• \"mostrame mis recordatorios\"\n"
        "• \"cuántas tareas hice?\"\n"
        "━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Hablame como a una persona, yo entiendo 😊"
    )

    await update.message.reply_text(help_text, parse_mode="Markdown")


async def reminders_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Mostrar todos los recordatorios programados."""
    uid = str(update.effective_user.id)
    text = chat.list_reminders(uid)
    await update.message.reply_text(text, parse_mode="Markdown")


async def delete_reminder_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Eliminar un recordatorio por número: /delete_reminder N"""
    uid = str(update.effective_user.id)

    if not context.args:
        await update.message.reply_text(
            "Decime qué recordatorio borrar. Ejemplo:\n"
            "/delete_reminder 1",
        )
        return

    try:
        index = int(context.args[0])
    except ValueError:
        await update.message.reply_text(
            "Pasame un número válido. Ejemplo:\n"
            "/delete_reminder 1",
        )
        return

    text = chat.delete_reminder(uid, index)
    await update.message.reply_text(text, parse_mode="Markdown")


async def delete_all_reminders_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Eliminar todos los recordatorios."""
    uid = str(update.effective_user.id)
    text = chat.delete_all_reminders(uid)
    await update.message.reply_text(text, parse_mode="Markdown")


async def add_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Crear una tarea usando /add.
    Ejemplo: /add Hacer el trabajo de sistemas inteligentes
    """
    uid = str(update.effective_user.id)

    if not context.args:
        await update.message.reply_text(
            "Decime qué querés agregar. Ejemplo:\n"
            "/add Hacer el trabajo de sistemas inteligentes",
        )
        return

    raw_text = " ".join(context.args)
    response = chat.add_task_smart(uid, raw_text, {})
    await update.message.reply_text(response, parse_mode="Markdown")


async def suggestion_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Sugiere un orden de prioridad para las tareas de hoy."""
    uid = str(update.effective_user.id)
    text = chat.suggest_task_order(uid)
    await update.message.reply_text(text)


# ------------------------------
# MAIN
# ------------------------------
def main():
    load_dotenv()
    token = os.getenv("TELEGRAM_TOKEN")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    app = Application.builder().token(token).build()

    # Comandos básicos
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    
    # Comandos de tareas
    app.add_handler(CommandHandler("tasks", tasks))
    app.add_handler(CommandHandler("add", add_command))
    app.add_handler(CommandHandler("done", done))
    app.add_handler(CommandHandler("suggestion", suggestion_command))
    
    # Comandos de recordatorios
    app.add_handler(CommandHandler("reminders", reminders_command))
    app.add_handler(CommandHandler("delete_reminder", delete_reminder_command))
    app.add_handler(CommandHandler("delete_all_reminders", delete_all_reminders_command))
    
    # Comandos de estadísticas
    app.add_handler(CommandHandler("today", today))
    app.add_handler(CommandHandler("stats", stats))

    # Handler de texto libre
    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_text,
        )
    )

    # Notificaciones de recordatorios (cada 60 segundos)
    app.job_queue.run_repeating(
        notify_due_reminders,
        interval=60,
        first=15,
    )

    logging.info("🤖 Bot inteligente iniciado")
  
    app.run_polling()


if __name__ == "__main__":
    main()