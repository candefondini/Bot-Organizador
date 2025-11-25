import os
from dotenv import load_dotenv

# Carga las variables definidas en el archivo .env al entorno del sistema
load_dotenv()

class Config:
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    # Modelos recomendados de Groq (todos gratuitos):
    # - llama-3.3-70b-versatile (el mejor, más inteligente)
    # - llama-3.1-70b-versatile (alternativa robusta)
    # - mixtral-8x7b-32768 (bueno para contextos largos)
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # Ruta del archivo donde se guarda la "base de datos" en formato JSON
    DATA_PATH = os.getenv("DATA_PATH", "data/conversation_history.json")