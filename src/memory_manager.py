import sqlite3
import logging
from datetime import datetime
from typing import Optional, List
from uuid import uuid4
from encoder_decoder import encode_text_local, decode_text_local
from config import model
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

DB_PATH = 'hack_memory.db'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - Chat %(chat_id)s - %(message)s',
    handlers=[
        logging.FileHandler('hack_memory.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimpleMemoryManager:
    """Lightweight SQLite store using LZMA encoder. Per-chat isolation + thresholds + grader + semantic recall + history."""
    def __init__(self, db_path: str = DB_PATH, token_limit: int = 8000, 
                 min_tokens_threshold: int = 50,
                 grade_threshold: int = 6,
                 use_grader: bool = False,
                 reabsorb_interval: int = 3,
                 enable_history: bool = True):
        self.db_path = db_path
        self.token_limit = token_limit
        self.min_tokens_threshold = min_tokens_threshold
        self.grade_threshold = grade_threshold
        self.use_grader = use_grader
        self.reabsorb_interval = reabsorb_interval
        self.turn_counter = 0
        self.chat_id = 'GLOBAL'
        self.enable_history = enable_history
        self.first_prompt = None  # For auto-title
        self.metrics = {'stores': 0, 'skipped_short': 0, 'skipped_low_grade': 0, 
                        'reabsorbs': 0, 'evictions': 0, 'total_compressed_chars': 0,
                        'semantic_recalls': 0}
        self.stored_texts = []
        self.vectorizer = TfidfVectorizer(max_features=100)
        self._init_db()
        logger.info("MemoryManager initialized", extra={'chat_id': self.chat_id})
    
    def set_chat_id(self, chat_id: str):
        self.chat_id = chat_id
        self.turn_counter = 0  # Reset for new session
        self.stored_texts = []  # Clear cache; rebuild on load
        if self.enable_history:
            self.create_session("New Chat")  # Ensure entry
        logger.info(f"Switched to chat session: {chat_id}", extra={'chat_id': self.chat_id})
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Encoded memory table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS encoded_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT,
                timestamp REAL,
                title TEXT,
                encoded_data TEXT,
                orig_tokens INTEGER
            )
        ''')
        
        # Migration for chat_id
        cursor.execute("PRAGMA table_info(encoded_memory)")
        columns = [row[1] for row in cursor.fetchall()]
        if 'chat_id' not in columns:
            cursor.execute('ALTER TABLE encoded_memory ADD COLUMN chat_id TEXT')
            logger.info("🔧 Migrated DB: Added chat_id column", extra={'chat_id': self.chat_id})
            conn.commit()
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chat_timestamp ON encoded_memory(chat_id, timestamp)')
        
        # Chat sessions table
        if self.enable_history:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    chat_id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at REAL,
                    last_updated REAL,
                    preview TEXT
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_last_updated ON chat_sessions(last_updated)')
        
        conn.commit()
        conn.close()
        logger.info("DB initialized/migrated successfully", extra={'chat_id': self.chat_id})
    
    def create_session(self, title: str = "New Chat") -> str:
        if not self.enable_history:
            return str(uuid4())
        if self.chat_id == 'GLOBAL':
            self.chat_id = str(uuid4())
        now = datetime.now().timestamp()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'INSERT OR IGNORE INTO chat_sessions (chat_id, title, created_at, last_updated, preview) VALUES (?, ?, ?, ?, ?)',
            (self.chat_id, title, now, now, "Welcome to chat!")
        )
        conn.commit()
        conn.close()
        logger.info(f"Created session: {self.chat_id} - {title}", extra={'chat_id': self.chat_id})
        return self.chat_id
    
    def update_session(self, title: str, preview: str):
        if not self.enable_history:
            return
        now = datetime.now().timestamp()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'UPDATE chat_sessions SET title = ?, last_updated = ?, preview = ? WHERE chat_id = ?',
            (title, now, preview, self.chat_id)
        )
        conn.commit()
        conn.close()
        logger.info(f"Updated session {self.chat_id}: {title}", extra={'chat_id': self.chat_id})
    
    def get_sessions(self) -> List[dict]:
        if not self.enable_history:
            return []
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT chat_id, title, created_at, last_updated, preview FROM chat_sessions ORDER BY last_updated DESC')
        rows = cursor.fetchall()
        conn.close()
        return [
            {'id': r[0], 'title': r[1], 'created': datetime.fromtimestamp(r[2]), 'updated': datetime.fromtimestamp(r[3]), 'preview': r[4]}
            for r in rows
        ]
    
    def get_session_history(self, chat_id: str) -> List[dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT title, orig_tokens, timestamp FROM encoded_memory 
            WHERE chat_id = ? ORDER BY timestamp ASC
        ''', (chat_id,))
        rows = cursor.fetchall()
        conn.close()
        return [{'title': r[0], 'tokens': r[1], 'time': datetime.fromtimestamp(r[2])} for r in rows]
    
    def delete_session(self, chat_id: str):
        if not self.enable_history:
            return
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM chat_sessions WHERE chat_id = ?', (chat_id,))
        cursor.execute('DELETE FROM encoded_memory WHERE chat_id = ?', (chat_id,))
        conn.commit()
        conn.close()
        logger.info(f"Deleted session {chat_id}", extra={'chat_id': 'GLOBAL'})
    
    def get_stored_tokens_for_session(self, chat_id: str) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT SUM(orig_tokens) FROM encoded_memory WHERE chat_id = ?', (chat_id,))
        total = cursor.fetchone()[0] or 0
        conn.close()
        return int(total)
    
    def _get_stored_raw(self, chat_id: str) -> List[tuple]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT encoded_data FROM encoded_memory WHERE chat_id = ? ORDER BY timestamp ASC', (chat_id,))
        raw_rows = cursor.fetchall()
        conn.close()
        return [(row[0], decode_text_local(row[0])) for row in raw_rows]
    
    def _grade_response(self, response):
        if not self.use_grader:
            return 7  # Default good
        length_score = min(10, len(response.split()) // 10)
        keyword_score = 5 if any(word in response.lower() for word in ['explain', 'detail', 'example']) else 3
        return (length_score + keyword_score) // 2
        
    def store_response(self, response: str, title: str, first_prompt: str = None, tokens: int = None):
        if first_prompt:
            self.first_prompt = first_prompt
        
        if tokens is None:
            tokens = len(response.split())
        
        if tokens < self.min_tokens_threshold:
            self.metrics['skipped_short'] += 1
            logger.info(f"Skipped short response '{title[:30]}...': {tokens} tokens (< {self.min_tokens_threshold})", extra={'chat_id': self.chat_id})
            return
        
        score = self._grade_response(response)
        if score < self.grade_threshold:
            self.metrics['skipped_low_grade'] += 1
            logger.info(f"Skipped low-grade response '{title[:30]}...': Score {score}/10", extra={'chat_id': self.chat_id})
            return
        
        # Auto-title on first store
        if self.first_prompt and 'New Chat' in title:
            auto_title = f"Chat about {self.first_prompt[:30].replace(' ', '_')}"
            self.update_session(auto_title, f"Started with: {self.first_prompt[:50]}...")
        
        encoded_dict = encode_text_local(response, title)
        orig_chars = len(response.encode('utf-8'))
        self.metrics['total_compressed_chars'] += orig_chars - len(encoded_dict['encoded_data'])
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO encoded_memory (chat_id, timestamp, title, encoded_data, orig_tokens) VALUES (?, ?, ?, ?, ?)',
            (self.chat_id, datetime.now().timestamp(), title, encoded_dict['encoded_data'], tokens)
        )
        conn.commit()
        conn.close()
        self.metrics['stores'] += 1
        ratio = orig_chars / len(encoded_dict['encoded_data']) if encoded_dict['encoded_data'] else 1
        logger.info(f"Stored '{title[:30]}...': {tokens} tokens (Grade: {score}/10) → {len(encoded_dict['encoded_data'])} chars ({ratio:.1f}x savings!)", extra={'chat_id': self.chat_id})
        
        # Update session preview
        if self.enable_history:
            preview = f"Last: {response[:50]}..."
            self.update_session(title if 'New Chat' not in title else title, preview)
        
        # Cache for semantic
        self.stored_texts.append((self.chat_id, response))
    
    def find_relevant(self, query: str, top_k: int = 2, min_sim: float = 0.3) -> List[str]:
        chat_texts = [text for cid, text in self.stored_texts if cid == self.chat_id]
        if len(chat_texts) < 2:
            logger.debug(f"No texts for semantic recall in chat {self.chat_id}", extra={'chat_id': self.chat_id})
            return []
        
        try:
            vectors = self.vectorizer.fit_transform(chat_texts + [query])
            sims = cosine_similarity(vectors[-1:], vectors[:-1])[0]
            top_idx = np.argsort(sims)[-top_k:][::-1]
            relevant = [chat_texts[i] for i in top_idx if sims[i] >= min_sim]
            self.metrics['semantic_recalls'] += 1
            logger.info(f"Found {len(relevant)} relevant memories for '{query[:30]}...' (top sims: {sims[top_idx[:len(relevant)]]})", extra={'chat_id': self.chat_id})
            return relevant
        except Exception as e:
            logger.warning(f"Semantic recall failed: {e}, fallback to []", extra={'chat_id': self.chat_id})
            return []
    
    def should_reabsorb(self, current_tokens: int) -> bool:
        self.turn_counter += 1
        high_usage = current_tokens > self.token_limit * 0.8
        on_interval = self.turn_counter % self.reabsorb_interval == 0
        decision = high_usage or on_interval
        logger.debug(f"Reabsorb check: Turn {self.turn_counter}, usage {current_tokens}/{self.token_limit}, decision: {decision}", extra={'chat_id': self.chat_id})
        return decision
    
    def reabsorb_oldest(self) -> str:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, encoded_data, orig_tokens FROM encoded_memory 
            WHERE chat_id = ? ORDER BY timestamp ASC LIMIT 1
        ''', (self.chat_id,))
        row = cursor.fetchone()
        if row:
            entry_id, encoded_data, tokens = row
            reabsorbed = decode_text_local(encoded_data)
            cursor.execute('DELETE FROM encoded_memory WHERE id = ? AND chat_id = ?', (entry_id, self.chat_id))
            conn.commit()
            conn.close()
            self.metrics['reabsorbs'] += 1
            self.metrics['evictions'] += 1
            logger.info(f"Reabsorbed {tokens} tokens from oldest (chat {self.chat_id}): '{reabsorbed[:50]}...'", extra={'chat_id': self.chat_id})
            return reabsorbed
        conn.close()
        logger.debug(f"No memories to reabsorb for chat {self.chat_id}", extra={'chat_id': self.chat_id})
        return ""
    
    def get_stored_tokens(self) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT SUM(orig_tokens) FROM encoded_memory WHERE chat_id = ?', (self.chat_id,))
        total = cursor.fetchone()[0] or 0
        conn.close()
        return int(total)
    
    def print_summary(self):
        total_tokens = self.get_stored_tokens()
        expansion = total_tokens / self.token_limit if total_tokens else 1
        skips = self.metrics['skipped_short'] + self.metrics['skipped_low_grade']
        
        logger.info(f"Session summary for {self.chat_id}: {self.metrics['stores']} stores ({skips} skipped), {self.metrics['reabsorbs']} reabsorbs, {self.metrics['semantic_recalls']} recalls, {expansion:.1f}x expansion! Saved {self.metrics['total_compressed_chars']} chars.", extra={'chat_id': self.chat_id})
        print(f"\n🎉 Chat {self.chat_id[:8]}... Metrics: {self.metrics['stores']} stores ({skips} skipped), {self.metrics['reabsorbs']} reabsorbs, {self.metrics['semantic_recalls']} recalls, {expansion:.1f}x expansion! Saved {self.metrics['total_compressed_chars']} chars.")
        
        # Viz: Bar chart
        fig, ax = plt.subplots(figsize=(8, 5))
        metrics = ['Stores', 'Skips', 'Reabsorbs', 'Recalls', 'Expansion (x)']
        values = [self.metrics['stores'], skips, self.metrics['reabsorbs'], self.metrics['semantic_recalls'], expansion]
        ax.bar(metrics, values, color=['blue', 'orange', 'green', 'purple', 'red'])
        ax.set_title(f"Chat {self.chat_id[:8]}... Impact")
        ax.set_ylabel('Value')
        plt.xticks(rotation=45)
        chart_path = f'metrics_{self.chat_id[:8]}.png'
        plt.savefig(chart_path)
        plt.close()
        logger.info(f"Metrics chart saved to {chart_path}", extra={'chat_id': self.chat_id})
        print(f"📊 Chart saved: {chart_path}")