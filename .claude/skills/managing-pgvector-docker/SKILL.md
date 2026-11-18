---
name: managing-pgvector-docker
description: Sets up, resets, backs up and debugs the PostgreSQL + pgvector Docker container whose data lives in the postgres-data folder. Use when starting the database, changing its password or port, moving or backing up its data, or when a database connection fails.
---

# Managing the pgvector Docker database

Reference implementation: `docker-compose.yml`, `.env.example`, `rag_lab/db/connection.py`.
The project runs PostgreSQL **only in Docker** (`pgvector/pgvector:pg17`). Windows-specific connection errors are covered by the local `troubleshooting-windows-env` skill when it exists (it is not part of the repository).
Helper script: [pgvector_docker_tools.py](pgvector_docker_tools.py) - compose template, password change, backup/restore, row count (`python pgvector_docker_tools.py` for usage).

## Set up in a new project

Copy this checklist and tick it off:

```
- [ ] 1. Copy docker-compose.yml (bind mount ./postgres-data, port ${POSTGRES_PORT}:5432, pg_isready healthcheck)
- [ ] 2. Add postgres-data/ to .gitignore
- [ ] 3. Create .env from .env.example; choose a free POSTGRES_PORT
- [ ] 4. docker compose up -d --wait
- [ ] 5. Connect with load_settings() and check pgvector
```

- Step 1: keep the data folder outside `data/` - the loader reads every `.txt`/`.md`/`.pdf` below `data/`.
- Step 3: if a locally installed PostgreSQL already listens on 5432, use another port such as 5433 - otherwise connections reach the wrong server.
- Step 4: on the very first start the server restarts once after `initdb` while the healthcheck already reports healthy. If the first connection fails, wait a few seconds and connect again.
- Step 5 passes when `SELECT extversion FROM pg_extension WHERE extname='vector'` returns a version (after the first ingestion). Otherwise return to step 3 and compare `.env` with the container.

## Change the password

`POSTGRES_PASSWORD` is applied only when `postgres-data/` is first created. After changing it in `.env`, run exactly this (Git Bash, project root) - it never prints the secret:

```bash
PW=$(uv run python -c "from dotenv import dotenv_values; print(dotenv_values('.env')['POSTGRES_PASSWORD'], end='')")
echo "ALTER USER postgres PASSWORD :'pw';" | docker exec -i basic-rag-stuff-postgres-1 psql -U postgres -v pw="$PW" -q
```

## Back up, restore, reset

Run exactly these commands; `MSYS_NO_PATHCONV=1` stops Git Bash from rewriting `/tmp/...` into a Windows path:

```bash
export MSYS_NO_PATHCONV=1
docker exec basic-rag-stuff-postgres-1 pg_dump -U postgres -d rag_learning --format=custom -f /tmp/rag_learning.dump
docker cp basic-rag-stuff-postgres-1:/tmp/rag_learning.dump ./rag_learning.dump
# restore into an empty database:
docker cp ./rag_learning.dump basic-rag-stuff-postgres-1:/tmp/rag_learning.dump
docker exec basic-rag-stuff-postgres-1 pg_restore -U postgres -d rag_learning --clean --if-exists /tmp/rag_learning.dump
```

- **Reset**: `docker compose down`, delete `postgres-data/`, `docker compose up -d`. (`down -v` deletes nothing here.)
- **Changing the volume setup**: first prove the new mount in a throwaway container, back up, switch, restore, then `docker compose down` + `up` and compare row counts with the backup. Proceed only when they match.

## Reference

- `"5433:5432"` binds all network interfaces; `"127.0.0.1:5433:5432"` limits access to this PC.
- Integration tests write to their own `test_documents` table; the `documents` table is never touched by tests.
