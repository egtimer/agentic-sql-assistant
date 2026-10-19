"""Entry point: ``python -m src.mcp_server [--transport stdio|http] ...``."""

import sys
from collections.abc import Sequence

from src.mcp_server.catalog import Catalog
from src.mcp_server.config import LOOPBACK_HOSTS, ServerConfig, parse_config
from src.mcp_server.server import create_server
from src.mcp_server.service import Service
from src.observability import tracing


def log(message: str) -> None:
    """Startup messages go to stderr: on stdio, stdout belongs to the protocol."""
    print(message, file=sys.stderr, flush=True)


def build_service(config: ServerConfig) -> Service:
    catalog = Catalog.scan(config.databases_dir)
    if not len(catalog):
        raise SystemExit(
            f"error: no *.sqlite or *.db files in {config.databases_dir}. For the Chinook "
            "demo run `make chinook-data`, or pass --databases-dir."
        )
    return Service(
        catalog,
        llm_config=config.llm_config,
        budget_usd=config.budget_usd,
        state_dir=config.state_dir,
    )


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_config(argv)
    service = build_service(config)
    server = create_server(service)
    budget = "none" if config.budget_usd is None else f"${config.budget_usd:.2f}"
    log(
        f"sql-agent MCP server: databases {', '.join(service.catalog.names)}; "
        f"model {config.llm_config.model}; budget {budget}; transport {config.transport}"
    )
    try:
        if config.transport == "stdio":
            server.run("stdio")
        else:
            if config.host not in LOOPBACK_HOSTS:
                log(
                    f"warning: {config.host} is not a loopback address and the HTTP transport "
                    "has no authentication: anyone who can reach it can use the model budget"
                )
            server.run("streamable-http", host=config.host, port=config.port)
    except KeyboardInterrupt:
        pass
    finally:
        tracing.flush()
        service.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
