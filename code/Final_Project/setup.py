import os
import google.generativeai as genai
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()

# ✅ Read API Keys from .env
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# ✅ Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

##### ✅ Debugging
print(f"Google API Key Loaded: {bool(GOOGLE_API_KEY)}")
print(f"Telegram Bot Token Loaded: {bool(TELEGRAM_BOT_TOKEN)}")

