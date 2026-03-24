import sqlite3
import json
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

DB_PATH = "face_db.sqlite"

def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            embedding TEXT NOT NULL
        )
    ''')
    
    # Create logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            name TEXT NOT NULL,
            status TEXT NOT NULL
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully.")

def register_user(name: str, embedding: list) -> bool:
    """Register a new user with their face embedding."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Convert embedding list to JSON string for storage
        embedding_json = json.dumps(embedding)
        
        cursor.execute(
            "INSERT INTO users (name, embedding) VALUES (?, ?)",
            (name, embedding_json)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        logger.error(f"User {name} already exists.")
        return False
    except Exception as e:
        logger.error(f"Error registering user: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def get_all_users() -> list:
    """Retrieve all users and their embeddings."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name, embedding FROM users")
        rows = cursor.fetchall()
        
        users = []
        for row in rows:
            name, embedding_json = row
            embedding = json.loads(embedding_json)
            users.append({"name": name, "embedding": embedding})
            
        return users
    except Exception as e:
        logger.error(f"Error fetching users: {e}")
        return []
    finally:
        if 'conn' in locals():
            conn.close()

def log_event(name: str, status: str):
    """Log an access attempt with spoof status."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO logs (name, status) VALUES (?, ?)",
            (name, status)
        )
        conn.commit()
    except Exception as e:
        logger.error(f"Error logging event: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

def get_recent_logs(limit: int = 50) -> list:
    """Fetch recent logs ordered by newest first."""
    try:
        conn = get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT timestamp, name, status FROM logs ORDER BY timestamp DESC LIMIT ?", 
            (limit,)
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Error fetching logs: {e}")
        return []
    finally:
        if 'conn' in locals():
            conn.close()

# Initialize the database if run directly
if __name__ == "__main__":
    init_db()
