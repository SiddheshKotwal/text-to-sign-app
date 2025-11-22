import sqlite3
import hashlib
from datetime import datetime

DB_NAME = "app_data.db"

def init_db():
    """Initialize the database with users and feedback tables."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Create Users Table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT)''')
    # Create Feedback Table
    c.execute('''CREATE TABLE IF NOT EXISTS feedback
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  username TEXT,
                  input_text TEXT, 
                  rating TEXT, 
                  comments TEXT,
                  timestamp TEXT)''')
    conn.commit()
    conn.close()

def create_user(username, password):
    """Create a new user (simple hash)."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False # User already exists
    finally:
        conn.close()

def check_login(username, password):
    """Verify credentials."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, hashed_pw))
    result = c.fetchone()
    conn.close()
    return result is not None

def delete_account(username):
    """
    Ethical Feature: Right to be forgotten.
    Deletes User AND their associated feedback data.
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # 1. Delete User
    c.execute("DELETE FROM users WHERE username=?", (username,))
    
    # 2. Delete User's Feedback (Ethical Update)
    c.execute("DELETE FROM feedback WHERE username=?", (username,))
    
    conn.commit()
    conn.close()

def save_feedback(username, text, rating, comments=""):
    """Ethical Feature: Human-in-the-loop feedback."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO feedback (username, input_text, rating, comments, timestamp) VALUES (?, ?, ?, ?, ?)", 
              (username, text, rating, comments, timestamp))
    conn.commit()
    conn.close()