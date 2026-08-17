import sqlite3
from pathlib import Path

# Create database inside banking_agent/
DB_PATH = Path(__file__).parent / "banking.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Create customers table
cursor.execute("""
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    age INTEGER,
    city TEXT,
    state TEXT,
    segment TEXT,
    risk_level TEXT
)
""")

# Sample customer data
customers = [
    ("CUST-100", "John Smith", 42, "Phoenix", "AZ", "Premier", "LOW"),
    ("CUST-101", "Sarah Lee", 35, "Seattle", "WA", "Standard", "LOW"),
    ("CUST-102", "Michael Brown", 51, "Denver", "CO", "Premier", "MEDIUM"),
    ("CUST-103", "Emily Davis", 29, "Austin", "TX", "Standard", "LOW"),
    ("CUST-104", "David Wilson", 47, "Chicago", "IL", "Premier", "HIGH"),
    ("CUST-105", "Lisa Anderson", 38, "Boston", "MA", "Standard", "LOW"),
    ("CUST-106", "Robert Taylor", 56, "Miami", "FL", "Premier", "MEDIUM"),
    ("CUST-107", "Jennifer Martinez", 33, "San Diego", "CA", "Standard", "LOW"),
    ("CUST-108", "Daniel Thomas", 45, "New York", "NY", "Premier", "HIGH"),
    ("CUST-109", "Amanda Jackson", 40, "Portland", "OR", "Standard", "LOW")
]

cursor.executemany("""
INSERT OR REPLACE INTO customers
(customer_id, name, age, city, state, segment, risk_level)
VALUES (?, ?, ?, ?, ?, ?, ?)
""", customers)

conn.commit()
conn.close()

print("Database created successfully!")
print("10 customers inserted.")