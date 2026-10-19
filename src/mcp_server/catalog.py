"""The databases a server exposes: an allowlist of names built once at startup.

Clients refer to a database by name only. A name is the file stem of a ``*.sqlite`` or
``*.db`` file found directly in the databases directory; the mapping to a path never
leaves the server, and nothing a client sends is ever turned into a path.
"""

from pathlib import Path

from src.sql_agent.db import connect_readonly

SUFFIXES = (".sqlite", ".db")


class UnknownDatabaseError(LookupError):
    """The client asked for a database that is not in the allowlist."""


class Catalog:
    """Name -> path of the databases of one server process."""

    def __init__(self, databases: dict[str, Path]) -> None:
        self._databases = dict(sorted(databases.items()))

    @classmethod
    def scan(cls, directory: Path) -> "Catalog":
        """Catalog of the ``*.sqlite`` and ``*.db`` files directly inside ``directory``.

        Raises ``ValueError`` when two files share a stem (the name would be ambiguous).
        """
        found: dict[str, Path] = {}
        if directory.is_dir():
            for path in sorted(directory.iterdir()):
                if path.is_file() and path.suffix.lower() in SUFFIXES:
                    if path.stem in found:
                        raise ValueError(
                            f"two database files share the name {path.stem!r} in {directory}"
                        )
                    found[path.stem] = path.resolve()
        return cls(found)

    @property
    def names(self) -> list[str]:
        return list(self._databases)

    def __len__(self) -> int:
        return len(self._databases)

    def path(self, name: str) -> Path:
        """Path of the database called ``name``; the error message lists the valid names."""
        try:
            return self._databases[name]
        except KeyError:
            valid = ", ".join(repr(n) for n in self._databases) or "(none)"
            raise UnknownDatabaseError(
                f"Unknown database {name!r}. Valid names: {valid}."
            ) from None

    def table_count(self, name: str) -> int:
        """Number of user tables of the database (read-only, trusted query)."""
        conn = connect_readonly(self.path(name))
        try:
            row = conn.execute(
                "SELECT count(*) FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            ).fetchone()
        finally:
            conn.close()
        return int(row[0])
