def portfolio_value(stocks):
    total = 0
    for stock in stocks:
        total += stock["shares"] * stock["price"]
    return total
