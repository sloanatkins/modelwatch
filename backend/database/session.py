"""
Database Session & Engine Setup
================================
This module configures the SQLite database connection,
creates the SQLAlchemy engine, session factory, and
initializes all database tables used by the application.

Responsibilities:
- Define database engine
- Provide DB session dependency
- Create tables on startup
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "sqlite:///./modelwatch.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()


def get_db():
    """
    FastAPI dependency that provides a database session.
    Ensures proper session cleanup after request.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
