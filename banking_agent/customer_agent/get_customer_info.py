import sqlite3
from pathlib import Path


def mask_ssn(ssn: str) -> str:
    """Masks SSN so only the last four digits are visible."""

    if not ssn:
        return "N/A"

    last_four = ssn[-4:]
    return f"***-**-{last_four}"


def get_customer_info(customer_id: str) -> dict:
    """Returns customer information with sensitive SSN data masked."""

    # banking.db lives one level above customer_agent/
    db_path = Path(__file__).parent.parent / "banking.db"

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT customer_id,
               name,
               age,
               city,
               state,
               segment,
               risk_level,
               ssn
        FROM customers
        WHERE customer_id = ?
        """,
        (customer_id,)
    )

    customer = cursor.fetchone()

    conn.close()

    if customer is None:
        return {
            "customer_id": customer_id,
            "status": "NOT_FOUND"
        }

    return {
        "customer_id": customer["customer_id"],
        "name": customer["name"],
        "age": customer["age"],
        "city": customer["city"],
        "state": customer["state"],
        "segment": customer["segment"],
        "risk_level": customer["risk_level"],
        "ssn": mask_ssn(customer["ssn"]),
        "status": "SUCCESS"
    }