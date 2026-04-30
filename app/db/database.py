from __future__ import annotations

import logging

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker


Base = declarative_base()
logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self, database_url: str):
        self._engine = create_engine(database_url, pool_pre_ping=True, future=True)
        self._session_factory = sessionmaker(bind=self._engine, autoflush=False, autocommit=False, expire_on_commit=False)

    @property
    def session_factory(self):
        return self._session_factory

    def ping(self) -> bool:
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def create_tables(self) -> None:
        from app.db import models  # noqa: F401

        Base.metadata.create_all(bind=self._engine)
        self._migrate_chat_schema()

    def _migrate_chat_schema(self) -> None:
        inspector = inspect(self._engine)
        with self._engine.begin() as conn:
            tables = set(inspector.get_table_names())
            if "chat_sessions" in tables:
                existing = {column["name"] for column in inspector.get_columns("chat_sessions")}
                if "summary_text" in existing:
                    conn.execute(text("UPDATE chat_sessions SET summary_text = COALESCE(summary_text, '')"))
                    try:
                        conn.execute(text("ALTER TABLE chat_sessions ALTER COLUMN summary_text SET DEFAULT ''"))
                    except Exception:
                        logger.debug("database.migration.summary_text_default_skipped", exc_info=True)
                if "title" not in existing:
                    conn.execute(text("ALTER TABLE chat_sessions ADD COLUMN title VARCHAR(200) NOT NULL DEFAULT '新会话'"))
                if "conversation_summary" not in existing:
                    conn.execute(text("ALTER TABLE chat_sessions ADD COLUMN conversation_summary TEXT NOT NULL DEFAULT ''"))
                if "summary_pending_chars" not in existing:
                    conn.execute(text("ALTER TABLE chat_sessions ADD COLUMN summary_pending_chars INTEGER NOT NULL DEFAULT 0"))
                if "summary_text" in existing:
                    conn.execute(text("UPDATE chat_sessions SET conversation_summary = COALESCE(conversation_summary, summary_text, '')"))
                else:
                    conn.execute(text("UPDATE chat_sessions SET conversation_summary = COALESCE(conversation_summary, '')"))
                try:
                    conn.execute(text("ALTER TABLE chat_sessions ALTER COLUMN conversation_summary SET DEFAULT ''"))
                except Exception:
                    logger.debug("database.migration.conversation_summary_default_skipped", exc_info=True)

            if "diagnostic_memories" in tables:
                existing = {column["name"] for column in inspector.get_columns("diagnostic_memories")}
                if "version_no" not in existing:
                    conn.execute(text("ALTER TABLE diagnostic_memories ADD COLUMN version_no INTEGER NOT NULL DEFAULT 1"))
                if "is_current" not in existing:
                    conn.execute(text("ALTER TABLE diagnostic_memories ADD COLUMN is_current VARCHAR(10) NOT NULL DEFAULT 'true'"))

    def dispose(self) -> None:
        self._engine.dispose()
