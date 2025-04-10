import os
from datetime import datetime, timezone
from typing import List

from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func, select

from src.core.agent import Message

# Get database URL from environment variables with a default for local development
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql+asyncpg://postgres:postgres@db:5432/agents"
)

# Create async engine
engine = create_async_engine(DATABASE_URL)
async_session_maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Define Base model
Base = declarative_base()


# Define ConversationMessage model
class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id = Column(Integer, primary_key=True)
    session_id = Column(String(36), index=True, nullable=False)
    message_order = Column(Integer, nullable=False)
    role = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


# Database initialization
async def init_db():
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("Database initialized successfully")
    except Exception as e:
        print(f"Error initializing database: {e}")
        # Don't crash the app if DB initialization fails, as we might be using
        # an in-memory store or a separate database service might be starting up


# Get database session
async def get_db():
    async with async_session_maker() as session:
        yield session


# Save a message to the database
async def save_message(db: AsyncSession, session_id: str, message: Message) -> None:
    try:
        # Determine the next message order
        max_order_query = select(func.max(ConversationMessage.message_order)).where(
            ConversationMessage.session_id == session_id
        )
        max_order_result = await db.execute(max_order_query)
        current_max_order = max_order_result.scalar_one_or_none()
        message_order = (current_max_order + 1) if current_max_order is not None else 0

        # Create new message
        db_message = ConversationMessage(
            session_id=session_id,
            message_order=message_order,
            role=message.role,
            content=message.content,
        )

        # Save to database
        db.add(db_message)
        await db.commit()
    except Exception as e:
        print(f"Error saving message: {e}")
        # Rollback in case of error
        await db.rollback()
        raise


# Get all messages for a session
async def get_messages_for_session(db: AsyncSession, session_id: str) -> List[Message]:
    try:
        query = (
            select(ConversationMessage)
            .where(ConversationMessage.session_id == session_id)
            .order_by(ConversationMessage.message_order)
        )

        result = await db.execute(query)
        db_messages = result.scalars().all()

        # Convert to Message objects
        messages = [Message(role=msg.role, content=msg.content) for msg in db_messages]

        return messages
    except Exception as e:
        print(f"Error retrieving messages for session {session_id}: {e}")
        return []  # Return empty list on error to avoid breaking the application flow


# Delete a conversation session
async def delete_session(db: AsyncSession, session_id: str) -> bool:
    try:
        query = ConversationMessage.__table__.delete().where(
            ConversationMessage.session_id == session_id
        )
        await db.execute(query)
        await db.commit()
        return True
    except Exception as e:
        print(f"Error deleting session {session_id}: {e}")
        await db.rollback()
        return False
