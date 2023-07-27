# pylint:disable=C0111,C0103
import sqlite3

conn = sqlite3.connect('data/ecommerce.sqlite')
db = conn.cursor()

def get_average_purchase(db):
    # return the average amount spent per order for each customer ordered by customer ID
    query = """WITH OrderValues AS (
            SELECT
                SUM(OrderDetails.UnitPrice * OrderDetails.Quantity) AS value,
                OrderDetails.OrderID
            FROM OrderDetails
            GROUP BY OrderDetails.OrderID
        )
        SELECT
            Customers.CustomerID,
            ROUND(AVG(OrderValues.value), 2) AS average
        FROM Customers
        JOIN Orders ON Customers.CustomerID = Orders.CustomerID
        JOIN OrderValues ON OrderValues.OrderID = Orders.OrderID
        GROUP BY Customers.CustomerID
        ORDER BY Customers.CustomerID
    """
    return db.execute(query).fetchall()

def get_general_avg_order(db):
    # return the average amount spent per order
    query = """
        WITH OrderValues AS (
            SELECT
                SUM(od.UnitPrice * od.Quantity) AS value
                FROM OrderDetails od
                GROUP BY od.OrderID
        )
        SELECT ROUND(AVG(ov.value), 2)
        FROM OrderValues ov
    """
    return db.execute(query).fetchone()[0]

def best_customers(db):
    # return the customers who have an average purchase greater than the general average purchase
    query = """
    WITH OrderValues AS (
        SELECT
            SUM(od.UnitPrice * od.Quantity) AS value,
            od.OrderID
        FROM OrderDetails od
        GROUP BY od.OrderID
        ),
        GeneralOrderValue AS (
            SELECT ROUND(AVG(ov.value), 2) AS average
            FROM OrderValues ov
        )
        SELECT
            c.CustomerID,
            ROUND(AVG(ov.value), 2) AS avg_amount_per_customer
        FROM Customers c
        JOIN Orders o ON c.CustomerID = o.CustomerID
        JOIN OrderValues ov ON o.OrderID = ov.OrderID
        GROUP BY c.CustomerID
        HAVING AVG(ov.value) > (SELECT average FROM GeneralOrderValue)
        ORDER BY avg_amount_per_customer DESC
        """
    return db.execute(query).fetchall()

def top_ordered_product_per_customer(db):
    # return the list of the top ordered product by each customer
    # based on the total ordered amount in USD
    query = """
    WITH OrderedProducts AS (
        SELECT o.CustomerID, od.ProductID, sum(od.Quantity * od.UnitPrice) AS ProductValue
        FROM OrderDetails od
        JOIN Orders o ON od.OrderID = o.OrderID
        GROUP BY o.CustomerID, od.ProductID
        ORDER BY ProductValue
        DESC
    )
    SELECT CustomerID, ProductID, MAX(ProductValue) AS TopProductValue
    FROM OrderedProducts
    GROUP BY CustomerID
    ORDER BY TopProductValue
    DESC
    """
    return db.execute(query).fetchall()

def average_number_of_days_between_orders(db):
    # return the average number of days between two consecutive orders of the same customer
    query = """
    WITH DatedOrders AS (
        SELECT
            CustomerID,
            OrderID,
            OrderDate,
            LAG(OrderDate, 1, 0) OVER (
                PARTITION BY CustomerID
                ORDER BY OrderDate
            ) PreviousOrderDate
        FROM Orders
        )
    SELECT ROUND(AVG(JULIANDAY(OrderDate) - JULIANDAY(PreviousOrderDate))) AS delta
    FROM DatedOrders
    WHERE PreviousOrderDate != 0
    """
    return int(db.execute(query).fetchone()[0])


print(average_number_of_days_between_orders(db))
