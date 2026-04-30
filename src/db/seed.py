"""Populate SQLite database with sample sales data for development and testing."""

import logging
import random
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "sample.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    region TEXT NOT NULL,
    tier TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sales (
    id INTEGER PRIMARY KEY,
    product_id INTEGER NOT NULL,
    customer_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    amount REAL NOT NULL,
    date TEXT NOT NULL,
    quarter TEXT NOT NULL,
    region TEXT NOT NULL,
    FOREIGN KEY (product_id) REFERENCES products(id),
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);
"""

PRODUCTS = [
    (1, "ProBook X1", "Electrónica", 899.00),
    (2, "ProBook X2", "Electrónica", 1299.00),
    (3, "TabletAir 10", "Electrónica", 549.00),
    (4, "Auriculares NoiseMax", "Electrónica", 199.00),
    (5, "Cargador Universal 100W", "Electrónica", 49.00),
    (6, "ErgoMax Pro", "Oficina", 459.00),
    (7, "Escritorio StandUp", "Oficina", 699.00),
    (8, "Monitor UltraWide 34", "Oficina", 579.00),
    (9, "Teclado MechPro", "Oficina", 129.00),
    (10, "LicenciaSuite Pro", "Software", 299.00),
]

CUSTOMERS = [
    (1, "TechCorp S.L.", "Norte", "Platinum"),
    (2, "Innovatech", "Norte", "Gold"),
    (3, "DataSystems", "Norte", "Silver"),
    (4, "CloudNine Solutions", "Norte", "Bronze"),
    (5, "MegaStore", "Centro", "Platinum"),
    (6, "DigitalHub", "Centro", "Gold"),
    (7, "StartupLab", "Centro", "Silver"),
    (8, "InfoServices", "Centro", "Bronze"),
    (9, "EuroTech GmbH", "Sur", "Gold"),
    (10, "MediterraneanIT", "Sur", "Gold"),
    (11, "SolTech", "Sur", "Silver"),
    (12, "CostaDigital", "Sur", "Silver"),
    (13, "NordSystems AB", "Norte", "Platinum"),
    (14, "AlpineSoft", "Centro", "Gold"),
    (15, "OceanData", "Sur", "Bronze"),
    (16, "UrbanTech", "Centro", "Silver"),
    (17, "PeakSolutions", "Norte", "Gold"),
    (18, "ValleyInnovations", "Sur", "Bronze"),
    (19, "RiverSoft", "Centro", "Silver"),
    (20, "SkyNet Services", "Norte", "Bronze"),
]

REGIONS = ["Norte", "Centro", "Sur"]

QUARTERS = {
    "Q1": ("2024-01-01", "2024-03-31"),
    "Q2": ("2024-04-01", "2024-06-30"),
    "Q3": ("2024-07-01", "2024-09-30"),
    "Q4": ("2024-10-01", "2024-12-31"),
}


def _random_date(start: str, end: str, rng: random.Random) -> str:
    """Generate a random date string between start and end (YYYY-MM-DD format)."""
    from datetime import datetime, timedelta

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    delta = (end_dt - start_dt).days
    random_days = rng.randint(0, delta)
    return (start_dt + timedelta(days=random_days)).strftime("%Y-%m-%d")


def _generate_sales(rng: random.Random) -> list[tuple[int, int, int, int, float, str, str, str]]:
    """Generate 150 sales records with realistic distribution.

    Sales volume grows each quarter to simulate business growth.
    Q3 target: ~€234,500 total to match golden dataset expectations.
    """
    sales = []
    sale_id = 1

    quarter_weights = {"Q1": 30, "Q2": 35, "Q3": 45, "Q4": 40}

    for quarter, (start_date, end_date) in QUARTERS.items():
        num_sales = quarter_weights[quarter]

        for _ in range(num_sales):
            product = rng.choice(PRODUCTS)
            product_id = product[0]
            price = product[3]
            customer = rng.choice(CUSTOMERS)
            customer_id = customer[0]
            region = customer[2]

            quantity = rng.randint(1, 15)
            amount = round(price * quantity * rng.uniform(0.90, 1.10), 2)

            date = _random_date(start_date, end_date, rng)

            sales.append((sale_id, product_id, customer_id, quantity, amount, date, quarter, region))
            sale_id += 1

    return sales


def seed_database(db_path: Path | None = None) -> Path:
    """Create and populate the SQLite database with sample data.

    Args:
        db_path: Path for the database file. Defaults to src/db/sample.db.

    Returns:
        Path to the created database file.
    """
    path = db_path or DB_PATH

    if path.exists():
        path.unlink()
        logger.info("Removed existing database at %s", path)

    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(SCHEMA)

        conn.executemany(
            "INSERT INTO products (id, name, category, price) VALUES (?, ?, ?, ?)",
            PRODUCTS,
        )

        conn.executemany(
            "INSERT INTO customers (id, name, region, tier) VALUES (?, ?, ?, ?)",
            CUSTOMERS,
        )

        rng = random.Random(42)
        sales = _generate_sales(rng)
        conn.executemany(
            "INSERT INTO sales (id, product_id, customer_id, quantity, amount, date, quarter, region) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            sales,
        )

        conn.commit()

        product_count = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        customer_count = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        sales_count = conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0]

        logger.info(
            "Database seeded: %d products, %d customers, %d sales",
            product_count,
            customer_count,
            sales_count,
        )

        for quarter in QUARTERS:
            total = conn.execute(
                "SELECT SUM(amount) FROM sales WHERE quarter = ?", (quarter,)
            ).fetchone()[0]
            logger.info("  %s total: €%.2f", quarter, total or 0)

    finally:
        conn.close()

    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    db = seed_database()
    print(f"Database created at: {db}")
