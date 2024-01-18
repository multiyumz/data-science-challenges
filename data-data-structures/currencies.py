# pylint: disable=missing-docstring

# TODO: add some currency rates
RATES = {"USDEUR": 0.85, "GBPEUR": 1.13, "CHFEUR": 0.86, "EURGBP": 0.885}

def convert(amount, currency):
    """returns the converted amount in the given currency
    amount is a tuple like (100, "EUR")
    currency is a string
    """

    # concat = amount[1] + currency
    # if concat in RATES.keys():
    #     return round(amount[0] * RATES[concat])
    # return None

    for key, value in RATES.items():
        if amount[1] == key[0:3] and currency == "EUR":
            return round(amount[0] * value)
        if amount[1] == key[3:6] and currency == key[0:3]:
            return round(amount[0] / value)



print(convert((100, "USD"), "EUR"))
