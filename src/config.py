import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get your Gemini API key
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("⚠️ GEMINI_API_KEY not found. Please add it to your .env file.")
else:
    print("✅ Gemini config loaded successfully!")
TOKEN_LIMIT = 2000
# Configure Gemini client
genai.configure(api_key=API_KEY)

# You can switch between gemini-1.5-pro or gemini-1.5-flash
MODEL_NAME = "gemini-2.5-flash"  # Updated to current; revert to "gemini-2.5-flash" if available

# Initialize the model instance for easy import
model = genai.GenerativeModel(MODEL_NAME)