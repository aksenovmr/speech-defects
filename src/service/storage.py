import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_DB_PATH = "artifacts/predictions.db"


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    ensure_parent_dir(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_column_exists(conn: sqlite3.Connection, table: str, column: str, column_type: str) -> None:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    existing_columns = {row["name"] for row in rows}
    if column not in existing_columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                request_id TEXT,
                session_id TEXT,
                filename TEXT,
                expected_sounds TEXT NOT NULL,
                p_bad REAL NOT NULL,
                is_bad INTEGER NOT NULL,
                status TEXT NOT NULL,
                flagged_sounds TEXT NOT NULL,
                all_sound_scores TEXT NOT NULL,
                model_name TEXT,
                model_version TEXT,
                thr_bad REAL,
                thr_sound REAL
            )
            """
        )

        _ensure_column_exists(conn, "predictions", "session_id", "TEXT")

        conn.commit()


def save_prediction(
    *,
    db_path: str = DEFAULT_DB_PATH,
    request_id: Optional[str],
    session_id: Optional[str],
    filename: str,
    expected_sounds: List[str],
    p_bad: float,
    is_bad: bool,
    status: str,
    flagged_sounds: List[str],
    all_sound_scores: Dict[str, Any],
    model_name: Optional[str],
    model_version: Optional[str],
    thr_bad: Optional[float],
    thr_sound: Optional[float],
) -> int:
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO predictions (
                request_id,
                session_id,
                filename,
                expected_sounds,
                p_bad,
                is_bad,
                status,
                flagged_sounds,
                all_sound_scores,
                model_name,
                model_version,
                thr_bad,
                thr_sound
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                session_id,
                filename,
                json.dumps(expected_sounds, ensure_ascii=False),
                float(p_bad),
                int(is_bad),
                status,
                json.dumps(flagged_sounds, ensure_ascii=False),
                json.dumps(all_sound_scores, ensure_ascii=False),
                model_name,
                model_version,
                thr_bad,
                thr_sound,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def list_predictions(
    *,
    db_path: str = DEFAULT_DB_PATH,
    limit: int = 100,
    session_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    with get_connection(db_path) as conn:
        if session_id:
            rows = conn.execute(
                """
                SELECT *
                FROM predictions
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT *
                FROM predictions
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    results: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["expected_sounds"] = json.loads(item["expected_sounds"])
        item["flagged_sounds"] = json.loads(item["flagged_sounds"])
        item["all_sound_scores"] = json.loads(item["all_sound_scores"])
        item["is_bad"] = bool(item["is_bad"])
        results.append(item)
    return results