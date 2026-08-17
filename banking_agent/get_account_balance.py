def get_account_balance(account_id: str) -> dict:
    """Returns the current balance for a given account."""

    balances = {
        "ACC-100": 5200,
        "ACC-200": 18500,
        "ACC-300": 750
    }

    balance = balances.get(account_id)

    if balance is None:
        return {
            "account_id": account_id,
            "status": "NOT_FOUND"
        }

    return {
        "account_id": account_id,
        "balance": balance,
        "currency": "USD",
        "status": "SUCCESS"
    }