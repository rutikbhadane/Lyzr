import streamlit as st
from uuid import uuid4
from config import model
from memory_manager import SimpleMemoryManager
from encoder_decoder import encode_text_local, decode_text_local

st.set_page_config(page_title="🧠 Dynamic Encoded Memory", layout="wide")
@st.cache_data
def estimate_tokens(history):
    return len(" ".join([f"{msg['role']}: {''.join([p['text'] for p in msg['parts']])}" for msg in history]).split())
def run_lzma_test():
    with st.sidebar.expander("🔐 LZMA Test"):
        sample_text = "The quick brown fox jumps over the lazy dog. This is a test for reversible encoding."
        if st.button("Run Test"):
            encoded = encode_text_local(sample_text, "sample_chunk")
            decoded = decode_text_local(encoded['encoded_data'])
            st.success("✅ Perfect match!" if decoded.strip() == sample_text.strip() else "⚠️ Check output.")
            st.text(f"Encoded (first 120): {encoded['encoded_data'][:120]}...")

@st.cache_resource
def get_manager():
    return SimpleMemoryManager(token_limit=8000, enable_history=True)

def main():
    st.title("🧠 Dynamic Encoded Memory Demo")
    st.markdown("Gemini + LZMA for infinite context. Chat, recall semantically, watch metrics!")
    
    if 'manager' not in st.session_state:
        st.session_state.manager = get_manager()
    
    manager = st.session_state.manager
    
    # Sidebar: History + Controls
    st.sidebar.title("📚 Chat History")
    sessions = manager.get_sessions()
    if sessions:
        for session in sessions:
            col1, col2 = st.columns([4, 1])
            with col1:
                # Clickable title: Button styled as link
                if st.button(f"**{session['title']}** ({session['updated'].strftime('%m/%d %H:%M')} | {manager.get_stored_tokens_for_session(session['id'])} tokens)", key=f"click_{session['id']}", help="Click to resume/open"):
                    st.session_state.chat_id = session['id']
                    manager.set_chat_id(st.session_state.chat_id)
                    # Rebuild cache
                    raw_stored = manager._get_stored_raw(st.session_state.chat_id)
                    manager.stored_texts = [(session['id'], text) for _, text in raw_stored]
                    # Reset history for resume
                    st.session_state.history = [{"role": "user", "parts": [{"text": f"Resuming session: {session['title']}. Continue from prior context."}]}]
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{session['id']}", help="Delete session"):
                    if 'chat_id' in st.session_state and st.session_state.chat_id == session['id']:
                        st.error("Can't delete active session!")
                    else:
                        manager.delete_session(session['id'])
                        st.rerun()
            st.sidebar.caption(session['preview'])
    else:
        st.sidebar.info("No past chats. Start one!")
    
    if st.sidebar.button("New Chat"):
        st.session_state.chat_id = manager.create_session("New Chat")
        manager.set_chat_id(st.session_state.chat_id)
        st.session_state.history = [{"role": "user", "parts": [{"text": "You are a helpful AI assistant. Keep responses concise but informative."}]}]
        st.rerun()
    
    # Current chat setup
    if 'chat_id' not in st.session_state:
        st.session_state.chat_id = manager.create_session("New Chat")
        manager.set_chat_id(st.session_state.chat_id)
        st.session_state.history = [{"role": "user", "parts": [{"text": "You are a helpful AI assistant. Keep responses concise but informative."}]}]
    
    chat_id = st.session_state.chat_id
    history = st.session_state.history
    
    # Chat interface
    for msg in history:
        role = "user" if msg["role"] == "user" else "assistant"
        st.chat_message(role).write(''.join([p['text'] for p in msg['parts']]))
    
    if prompt := st.chat_input("Ask Gemini (or 'recall [topic]' for semantic pull)..."):
        # Set first prompt for auto-title
        if manager.first_prompt is None:
            manager.first_prompt = prompt
        
        # Semantic recall
        recall_injected = ""
        if prompt.lower().startswith('recall '):
            query = prompt[7:].strip()
            relevant = manager.find_relevant(query)
            if relevant:
                recall_injected = "\n".join(relevant)
                history[0]['parts'][0]['text'] += f"\nRelevant prior context: {recall_injected}"
                st.chat_message("system").write(f"🔍 Recalled: {recall_injected[:200]}...")
            prompt = f"Recall query was '{query}', but continue conversation."
        
        # Token est.
        current_prompt = "\n".join([f"{msg['role']}: {''.join([p['text'] for p in msg['parts']])}" for msg in history])
        current_tokens = len(current_prompt.split())
        
        # Reabsorb
        if manager.should_reabsorb(current_tokens):
            reabsorbed = manager.reabsorb_oldest()
            if reabsorbed:
                history[0]['parts'][0]['text'] += f"\nReabsorbed prior context: {reabsorbed}"
                st.chat_message("system").write("📜 Context reabsorbed!")
        
        # Gemini
        history.append({"role": "user", "parts": [{"text": prompt}]})
        try:
            response = model.generate_content(history)
            assistant_text = response.text
        except Exception as e:
            assistant_text = f"Error: {e}"
        
        # Store with first_prompt for auto-title
        title = f"Turn {len(history)//2}: {prompt[:20]}..."
        manager.store_response(assistant_text, title, prompt)
        
        # Update & display
        history.append({"role": "model", "parts": [{"text": assistant_text}]})
        st.chat_message("user").write(prompt)
        st.chat_message("assistant").write(assistant_text)
        st.rerun()
    
    # Sidebar Metrics
    with st.sidebar.expander("📊 Metrics"):
        total_tokens = manager.get_stored_tokens()
        expansion = total_tokens / manager.token_limit if total_tokens else 1
        skips = manager.metrics['skipped_short'] + manager.metrics['skipped_low_grade']
        st.json({
            'Stores': manager.metrics['stores'],
            'Skips': skips,
            'Reabsorbs': manager.metrics['reabsorbs'],
            'Recalls': manager.metrics['semantic_recalls'],
            'Expansion (x)': f"{expansion:.1f}",
            'Saved Chars': manager.metrics['total_compressed_chars']
        })
        if st.button("Summary & Chart"):
            manager.print_summary()
    
    if st.sidebar.button("End Chat & Viz"):
        manager.print_summary()
        st.balloons()
    
    run_lzma_test()

if __name__ == "__main__":
    main()