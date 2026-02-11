from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, scoped_session
from sqlalchemy.exc import SQLAlchemyError
import logging
import json
import time
import os
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

Base = declarative_base()

class ConversationModel(Base):
    __tablename__ = 'conversations'
    
    # Composite primary key simulation or just use ID
    # For simplicity and compatibility with previous schema, we use composite key logic
    # But SQLAlchemy prefers single PK. We will use user_id + conversation_id as logical key
    # To handle composite PK in SQLAlchemy:
    user_id = Column(String, primary_key=True)
    conversation_id = Column(String, primary_key=True)
    data_json = Column(Text)
    updated_at = Column(String)

class DatabaseManager:
    """
    Manages persistence using SQLAlchemy (Supports SQLite & Postgres).
    """
    def __init__(self):
        # Default to SQLite if not set
        self.database_url = os.getenv("DATABASE_URL", "sqlite:///./chat_history.db")
        
        # Ensure SQLite uses absolute path logic if needed, but relative acts fine usually
        if self.database_url.startswith("sqlite"):
            connect_args = {"check_same_thread": False} # Needed for SQLite in FastAPI
        else:
            connect_args = {}

        try:
            self.engine = create_engine(self.database_url, connect_args=connect_args)
            self.SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=self.engine))
            
            # Create tables
            Base.metadata.create_all(bind=self.engine)
            logger.info(f"Database connected: {self.database_url}")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            self.SessionLocal = None

    def get_session(self):
        if not self.SessionLocal:
            return None
        return self.SessionLocal()

    def save_context(self, user_id: str, conversation_id: str, context_data: Dict[str, Any]):
        """Save or update conversation context."""
        session = self.get_session()
        if not session:
            return

        try:
            json_data = json.dumps(context_data)
            updated_at = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Check if exists
            existing = session.query(ConversationModel).filter_by(user_id=user_id, conversation_id=conversation_id).first()
            
            if existing:
                existing.data_json = json_data
                existing.updated_at = updated_at
            else:
                new_record = ConversationModel(
                    user_id=user_id, 
                    conversation_id=conversation_id, 
                    data_json=json_data, 
                    updated_at=updated_at
                )
                session.add(new_record)
            
            session.commit()
        except SQLAlchemyError as e:
            logger.error(f"DB Save Error: {e}")
            session.rollback()
        finally:
            session.close()

    def load_context(self, user_id: str, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Load conversation context."""
        session = self.get_session()
        if not session:
            return None

        try:
            record = session.query(ConversationModel).filter_by(user_id=user_id, conversation_id=conversation_id).first()
            if record:
                return json.loads(record.data_json)
            return None
        except SQLAlchemyError as e:
            logger.error(f"DB Load Error: {e}")
            return None
        finally:
            session.close()

    def get_recent_conversations(self, user_id: str, limit: int = 10):
        """Get list of recent conversations for a user."""
        session = self.get_session()
        if not session:
            return []
        try:
            # We need to import desc if we want dynamic ordering, or just use text
            from sqlalchemy import desc
            records = session.query(ConversationModel.conversation_id, ConversationModel.updated_at)\
                .filter_by(user_id=user_id)\
                .order_by(desc(ConversationModel.updated_at))\
                .limit(limit).all()
            return [{"id": r.conversation_id, "updated_at": r.updated_at} for r in records]
        except SQLAlchemyError as e:
            logger.error(f"DB List Error: {e}")
            return []
        finally:
            session.close()

