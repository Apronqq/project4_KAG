from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    __table_args__ = (UniqueConstraint("session_id", name="uq_chat_session_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)
    title: Mapped[str] = mapped_column(String(200), default="新会话", nullable=False)
    summary_text: Mapped[str] = mapped_column(Text, default="", nullable=False)
    conversation_summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    summary_pending_chars: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    conversation_memories = relationship("ConversationMemory", back_populates="session", cascade="all, delete-orphan")
    fact_memories = relationship("UserFactMemory", back_populates="session", cascade="all, delete-orphan")
    diagnostic_memories = relationship("DiagnosticMemory", back_populates="session", cascade="all, delete-orphan")


class ConversationMemory(Base):
    __tablename__ = "conversation_memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_ref_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    session = relationship("ChatSession", back_populates="conversation_memories")


class UserFactMemory(Base):
    __tablename__ = "user_fact_memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_ref_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    fact_group: Mapped[str] = mapped_column(String(50), nullable=False)
    fact_key: Mapped[str] = mapped_column(String(100), nullable=False)
    fact_value: Mapped[str] = mapped_column(Text, nullable=False)
    fact_unit: Mapped[str] = mapped_column(String(50), default="", nullable=False)
    source_label: Mapped[str] = mapped_column(String(50), default="pipeline", nullable=False)
    confidence: Mapped[str] = mapped_column(String(20), default="high", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    session = relationship("ChatSession", back_populates="fact_memories")


class DiagnosticMemory(Base):
    __tablename__ = "diagnostic_memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_ref_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    version_no: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    is_current: Mapped[str] = mapped_column(String(10), default="true", nullable=False)
    health_status: Mapped[str] = mapped_column(String(50), default="", nullable=False)
    urgency_level: Mapped[str] = mapped_column(String(50), default="", nullable=False)
    risk_summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    abnormal_indicator_summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    department_summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    follow_up_summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    lifestyle_summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    medication_summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    contraindication_summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    evidence_summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    session = relationship("ChatSession", back_populates="diagnostic_memories")
