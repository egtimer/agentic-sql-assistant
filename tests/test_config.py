"""Settings must load from a .env shared with docker compose."""

from pathlib import Path

from src.config import Settings


def test_env_file_with_compose_keys_loads(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "LANGFUSE_PUBLIC_KEY=pk-test\n"
        "LANGFUSE_HOST=http://localhost:3000\n"
        "LANGFUSE_DB_PASSWORD=x\n"
        "LANGFUSE_INIT_USER_EMAIL=dev@example.com\n"
        "LANGFUSE_ENCRYPTION_KEY=0000\n",
        encoding="utf-8",
    )
    s = Settings(_env_file=env)  # type: ignore[call-arg]
    assert s.langfuse_public_key == "pk-test"
    assert s.langfuse_host == "http://localhost:3000"
    assert not hasattr(s, "langfuse_db_password")


def test_example_env_file_loads() -> None:
    example = Path(__file__).resolve().parent.parent / ".env.example"
    Settings(_env_file=example)  # type: ignore[call-arg]
