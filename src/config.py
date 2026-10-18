from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    ollama_num_ctx: int | None = None
    embedding_model: str = "all-MiniLM-L6-v2"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "company_docs"

    sqlite_db_path: str = str(Path(__file__).parent / "db" / "sample.db")

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    sql_max_rows: int = 50
    sql_read_only: bool = True
    rag_top_k: int = 5

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
