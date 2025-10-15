from encoder_decoder import encode_text_local, decode_text_local
from chat_handler import chat_with_memory
from memory_manager import SimpleMemoryManager
from config import model  # Triggers config load

if __name__ == "__main__":
    TOKEN_LIMIT = 2000
    # Launch modular demo (new session each run)
    manager = SimpleMemoryManager(token_limit=TOKEN_LIMIT)
    chat_with_memory(manager)