from typing import List
from uuid import uuid4
from config import model
from memory_manager import SimpleMemoryManager
import logging 
TOKEN_LIMIT = 8000

def chat_with_memory(manager: SimpleMemoryManager):
    """Interactive chat loop: Gemini + interval reabsorption + semantic recall, per-chat isolation."""
    # New chat session
    chat_id = str(uuid4())
    manager.set_chat_id(chat_id)
    
    conversation_history = [{"role": "user", "parts": [{"text": "You are a helpful AI assistant. Keep responses concise but informative."}]}]
    print(f"🤖 New Chat Session: {chat_id[:8]}... | Gemini Boss LLM + LZMA Memory Demo (type 'exit' to quit)\n")
    print("💡 Tip: Say 'recall [topic]' to pull relevant memories semantically!\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'exit':
            manager.print_summary()
            break
        
        # Semantic recall trigger
        recall_injected = ""
        if user_input.lower().startswith('recall '):
            query = user_input[7:].strip()  # "recall ethics" -> "ethics"
            relevant = manager.find_relevant(query)
            if relevant:
                recall_injected = "\n".join(relevant)
                conversation_history[0]['parts'][0]['text'] += f"\nRelevant prior context: {recall_injected}"
                print(f"🔍 Recalled {len(relevant)} relevant items: '{recall_injected[:100]}...'")
            user_input = f"Recall query was '{query}', but continue conversation."  # Fallback prompt
        
        # Est. current tokens
        current_prompt = "\n".join([f"{msg['role']}: {''.join([p['text'] for p in msg['parts']])}" for msg in conversation_history])
        current_tokens = len(current_prompt.split())
        
        # Interval/usage-based reabsorb (isolated to chat_id)
        if manager.should_reabsorb(current_tokens):
            reabsorbed = manager.reabsorb_oldest()
            if reabsorbed:
                conversation_history[0]['parts'][0]['text'] += f"\nReabsorbed prior context: {reabsorbed}"
                print("📜 Context reabsorbed (interval/usage trigger)!")
        
        # Gemini call
        conversation_history.append({"role": "user", "parts": [{"text": user_input}]})
        try:
            response = model.generate_content(conversation_history)
            assistant_text = response.text
        except Exception as e:
            assistant_text = f"Error: {e}"
            logging.warning(f"Gemini API error: {e}")
        
        # Store (filtered + logged)
        title = f"Turn {len(conversation_history)//2}: {user_input[:20]}..."
        manager.store_response(assistant_text, title)
        
        # Update history
        conversation_history.append({"role": "model", "parts": [{"text": assistant_text}]})
        
        print(f"Gemini: {assistant_text}\n")