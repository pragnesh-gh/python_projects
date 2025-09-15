# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def last_transactions(user_transactions: list, transaction: str, max_length: int = 3):
    print(f"len(user_transactions) = {len(user_transactions)}, condition: {len(user_transactions) > max_length}")
    if len(user_transactions) >= max_length:
        user_transactions.pop(0)
    user_transactions.append(transaction)
    return user_transactions

def test_first_transaction_is_dropped():
    expected_value = 4
    transactions: list[str] = []
    transactions = last_transactions(transactions, "deposit $100")
    transactions = last_transactions(transactions, "deposit $200")
    transactions = last_transactions(transactions, "deposit $300")
    transactions = last_transactions(transactions, "deposit $400")

    assert(len(transactions) == expected_value), f"Expected {expected_value} but got {len(transactions)} transactions"
    assert(transactions[0] == "deposit $200"), f"Expected 'deposit $200' but got {len(transactions)} transactions"


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    transactions: list[str] = []
    transactions = last_transactions(transactions, "deposit $100")
    transactions = last_transactions(transactions, "deposit $200")
    transactions = last_transactions(transactions, "deposit $300")
    transactions = last_transactions(transactions, "deposit $400")
    print(transactions)

    test_first_transaction_is_dropped()


