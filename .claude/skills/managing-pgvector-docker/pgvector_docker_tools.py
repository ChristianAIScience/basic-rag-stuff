"""PostgreSQL + pgvector in Docker: compose template, password change, backup/restore, row count.

Dependencies:  Docker with the compose plugin; Python standard library only.

Usage (from the project root):
    python pgvector_docker_tools.py compose-template > docker-compose.yml
    python pgvector_docker_tools.py backup  rag_learning.dump
    python pgvector_docker_tools.py restore rag_learning.dump
    python pgvector_docker_tools.py count   documents
    python pgvector_docker_tools.py password             # reads POSTGRES_PASSWORD from .env, never prints it

Subprocess calls pass arguments as a list, so Git Bash path rewriting (/tmp -> C:/...) does not apply.
"""

import subprocess
import sys
from pathlib import Path

CONTAINER = "basic-rag-stuff-postgres-1"   # docker ps --format "{{.Names}}"
DATABASE = "rag_learning"
USER = "postgres"

COMPOSE_TEMPLATE = """\
services:
  postgres:
    image: pgvector/pgvector:pg17
    environment:
      POSTGRES_DB: ${POSTGRES_DB:-rag_learning}
      POSTGRES_USER: ${POSTGRES_USER:-postgres}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?Set POSTGRES_PASSWORD in .env}
    ports:
      - "${POSTGRES_PORT:-5432}:5432"      # 127.0.0.1:...:5432 to allow this PC only
    volumes:
      - ./postgres-data:/var/lib/postgresql/data   # keep it outside the documents folder
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-postgres}"]
      interval: 2s
      retries: 15
"""


def run(args: list[str], stdin: str | None = None) -> str:
    result = subprocess.run(args, input=stdin, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(args[:4])} ... failed: {result.stderr.strip()}")
    return result.stdout


def read_env_value(name: str, env_file: Path = Path(".env")) -> str:
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip()
    raise KeyError(f"{name} not found in {env_file}")


def change_password(new_password: str) -> None:
    """POSTGRES_PASSWORD only applies when the data folder is first created; this updates a running DB.
    The password goes through a psql variable on stdin - it never appears in the command line or output."""
    run(["docker", "exec", "-i", CONTAINER, "psql", "-U", USER, "-q", "-v", f"pw={new_password}"],
        stdin="ALTER USER postgres PASSWORD :'pw';")


def backup(out_file: Path) -> None:
    run(["docker", "exec", CONTAINER, "pg_dump", "-U", USER, "-d", DATABASE, "--format=custom", "-f", "/tmp/backup.dump"])
    run(["docker", "cp", f"{CONTAINER}:/tmp/backup.dump", str(out_file)])


def restore(dump_file: Path) -> None:
    run(["docker", "cp", str(dump_file), f"{CONTAINER}:/tmp/backup.dump"])
    run(["docker", "exec", CONTAINER, "pg_restore", "-U", USER, "-d", DATABASE, "--clean", "--if-exists", "/tmp/backup.dump"])


def count_rows(table: str) -> int:
    return int(run(["docker", "exec", CONTAINER, "psql", "-U", USER, "-d", DATABASE, "-tAc",
                    f'SELECT count(*) FROM "{table}"']).strip())


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "compose-template":
        print(COMPOSE_TEMPLATE, end="")
    elif command == "backup":
        backup(Path(sys.argv[2]))
    elif command == "restore":
        restore(Path(sys.argv[2]))
    elif command == "count":
        print(count_rows(sys.argv[2]))
    elif command == "password":
        change_password(read_env_value("POSTGRES_PASSWORD"))
        print("password updated")
    else:
        print(__doc__)
