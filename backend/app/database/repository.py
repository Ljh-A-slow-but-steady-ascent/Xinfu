from __future__ import annotations

from pathlib import Path
import os

from sqlalchemy import Engine, create_engine, inspect
from sqlalchemy.engine import make_url
from sqlalchemy.pool import StaticPool

from backend.app.database.models import Base


DEFAULT_DB = Path(__file__).resolve().parents[3] / "data" / "dermaguard.db"


def init_database(database_url: str | None = None) -> Engine:
    if database_url is None:
        database_url = os.getenv("DERMAGUARD_DATABASE_URL")
    if database_url is None:
        DEFAULT_DB.parent.mkdir(parents=True, exist_ok=True)
        database_url = f"sqlite:///{DEFAULT_DB}"
    url = make_url(database_url)
    if url.get_backend_name() == "sqlite" and url.database in (None, "", ":memory:"):
        # A worker thread must see the same ephemeral database as the ASGI
        # thread. File-backed SQLite retains its existing pool and options.
        engine = create_engine(database_url, poolclass=StaticPool,
                               connect_args={"check_same_thread": False})
    else:
        engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    # create_all does not add columns to an existing SQLite database.  Keep the
    # migration intentionally additive so older local cases remain readable.
    if engine.dialect.name == "sqlite":
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "UPDATE users SET full_name = ? WHERE full_name = ?",
                ("信肤管理员", "霁肤管理员"),
            )
        columns = {item["name"] for item in inspect(engine).get_columns("case_patients")}
        if "clinical_profile_json" not in columns:
            with engine.begin() as connection:
                connection.exec_driver_sql(
                    "ALTER TABLE case_patients "
                    "ADD COLUMN clinical_profile_json TEXT NOT NULL DEFAULT '{}'"
                )
    return engine
