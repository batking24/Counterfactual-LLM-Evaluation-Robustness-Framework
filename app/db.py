import sqlite3
import os
import json

DB_PATH = "outputs/logs/eval_logs.db"

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # For clean testing across iterations, drop old table
    cursor.execute('DROP TABLE IF EXISTS eval_runs')
    cursor.execute('''
        CREATE TABLE eval_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            answer TEXT,
            retrieved_context TEXT,
            grounding_score REAL,
            is_supported BOOLEAN,
            consistency_score REAL,
            hit_rate REAL,
            attribution_score REAL,
            hallucinated_claims INTEGER,
            safety_score REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def log_eval_run(record: dict):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO eval_runs 
        (query, answer, retrieved_context, grounding_score, is_supported, consistency_score, hit_rate, attribution_score, hallucinated_claims, safety_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        record.get("query"),
        record.get("answer"),
        json.dumps(record.get("retrieved_context", [])),
        record.get("grounding_score", 0.0),
        record.get("is_supported", False),
        record.get("consistency_score", 0.0),
        record.get("hit_rate", 0.0),
        record.get("attribution_score", 0.0),
        record.get("hallucinated_claims", 0),
        record.get("safety_score", 0.0)
    ))
    conn.commit()
    conn.close()
