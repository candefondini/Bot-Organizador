import os
from dotenv import load_dotenv

# Carga las variables definidas en el archivo .env al entorno del sistema
load_dotenv()

class Config:
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Ruta del archivo donde se guarda la "base de datos" en formato JSON
    DATA_PATH = os.getenv("DATA_PATH", "data/conversation_history.json")
