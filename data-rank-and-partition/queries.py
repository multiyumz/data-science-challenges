# pylint:disable=C0111,C0103
import sqlite3

conn = sqlite3.connect('data/ecommerce.sqlite')
db = conn.cursor()

def order_rank_per_customer(db):
    query = """
        SELECT
            OrderID,
            CustomerID,
            OrderDate,
            RANK() OVER (
                PARTITION BY CustomerID
                ORDER BY OrderDate
            ) OrderRank
        FROM Orders
        """
    orders = db.execute(query).fetchall()
    return orders


def order_cumulative_amount_per_customer(db):
    query = """
        SELECT
            Orders.OrderID,
            Orders.CustomerID,
            Orders.OrderDate,
            SUM(SUM(OrderDetails.UnitPrice * OrderDetails.Quantity)) OVER(PARTITION BY Orders.CustomerID ORDER BY Orders.OrderDate) AS OrderCumulativeAmount
        FROM Orders
        JOIN OrderDetails ON Orders.OrderID = OrderDetails.OrderID
        GROUP BY Orders.OrderID
        """
    orders = db.execute(query).fetchall()
    return orders

print(order_cumulative_amount_per_customer(db))
