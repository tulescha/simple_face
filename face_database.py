import sqlite3
import numpy as np

class FaceDatabase:
    def __init__(self, db_path="faces.db"):
        self.db_path = db_path
        self._create_table()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _create_table(self):
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    features BLOB NOT NULL
                )
            """)
            conn.commit()

    def add_face(self, name, features: np.ndarray):
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO faces (name, features) VALUES (?, ?)",
                           (name, features.tobytes()))
            conn.commit()

    def all_faces(self):
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, features FROM faces")
            rows = cursor.fetchall()
            return {name: np.frombuffer(f, dtype=np.float32) for name, f in rows}

    def has_faces(self):
        return len(self.all_faces()) > 0

    def delete_face(self, name):
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM faces WHERE name=?", (name,))
            conn.commit()
