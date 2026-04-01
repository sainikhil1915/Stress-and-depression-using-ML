import sqlite3

def init_db():
    # Connect to (or create) the database
    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    # Create 'users' table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL,
            password TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0
        )
    ''')

    # Create 'posts' table
    c.execute('''
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            prediction_svm TEXT,
            prediction_mlp TEXT,
            prediction_tfidf TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    conn.commit()
    conn.close()
    print("Database initialized successfully with users and posts tables.")


def insert_sample_users():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    # Sample users to insert
    users = [
        ("Alice Johnson", "alice", "alice@example.com", "password123", 0),
        ("Bob Smith", "bob", "bob@example.com", "password456", 0),
        ("Carol Admin", "admin", "admin@example.com", "admin", 1)
    ]

    for name, username, email, password, is_admin in users:
        try:
            c.execute('''
                INSERT INTO users (name, username, email, password, is_admin)
                VALUES (?, ?, ?, ?, ?)
            ''', (name, username, email, password, is_admin))
        except sqlite3.IntegrityError:
            print(f"User '{username}' already exists. Skipping...")

    conn.commit()
    conn.close()
    print("Sample users inserted.")


if __name__ == '__main__':
    init_db()
    insert_sample_users()
