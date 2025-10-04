#!/usr/bin/env python3
"""
Programmatic Alembic runner that:
- Avoids import shadowing by local 'alembic_migrations' folder.
- Converts async DATABASE_URL (asyncpg) to sync (psycopg2) for Alembic.
- Upgrades schema to head.
"""
import os
import sys
from pathlib import Path

# --- Safety: ensure we import the real Alembic package, not the local migrations dir
REPO_ROOT = Path(__file__).resolve().parent
local_shadow = REPO_ROOT / "alembic"
if local_shadow.exists():
    # Shouldn’t happen after renaming, but just in case:
    # Move cwd to end of sys.path so site-packages wins.
    try:
        sys.path.remove(str(REPO_ROOT))
    except ValueError:
        pass
    sys.path.append(str(REPO_ROOT))

# --- Build a sync URL for Alembic
db_url = (
    os.environ.get("DATABASE_URL") or os.environ.get("DJANGO_DATABASE_URL") or ""
).strip()

if not db_url:
    # Fallback from discrete parts (Compose env)
    user = os.environ.get("DB_USER", "mongars")
    pwd = os.environ.get("DB_PASSWORD", "changeme")
    host = os.environ.get("DB_HOST", "postgres")
    port = os.environ.get("DB_PORT", "5432")
    name = os.environ.get("DB_NAME", "mongars_db")
    db_url = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{name}"

# Convert asyncpg → psycopg2 for Alembic if needed
if db_url.startswith("postgresql+asyncpg://"):
    sync_url = db_url.replace("postgresql+asyncpg://", "postgresql+psycopg2://", 1)
else:
    sync_url = db_url

# --- Run Alembic
try:
    from alembic import command
    from alembic.config import Config
except Exception as e:
    print(
        "Failed to import alembic package. Is it installed in the image?",
        file=sys.stderr,
    )
    raise

alembic_ini = REPO_ROOT / "alembic.ini"
if not alembic_ini.exists():
    print(f"alembic.ini not found at {alembic_ini}", file=sys.stderr)
    sys.exit(2)

cfg = Config(str(alembic_ini))
# Force URL from env to avoid desync with alembic.ini
cfg.set_main_option("sqlalchemy.url", sync_url)

print(f"[init_db] Using SQLAlchemy URL: {sync_url}")
command.upgrade(cfg, "head")
print("[init_db] Alembic upgrade to head completed.")
