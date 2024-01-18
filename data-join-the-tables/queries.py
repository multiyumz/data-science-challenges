# pylint:disable=C0111,C0103

import sqlite3

conn = sqlite3.connect('data/ecommerce.sqlite')
db = conn.cursor()

def detailed_orders(db):
    '''return a list of all orders (order_id, customer.contact_name,
    employee.firstname) ordered by order_id'''
    query = """
        SELECT
                orders.OrderID,
                customers.ContactName,
                employees.FirstName
        FROM orders
        JOIN customers ON orders.customerID = customers.customerID
        JOIN employees ON orders.employeeID = employees.employeeID
        ORDER BY orders.OrderID
    """
    return db.execute(query).fetchall()


def spent_per_customer(db):
    '''return the total amount spent per customer ordered by ascending total
    amount (to 2 decimal places)
    Exemple :
        Jean   |   100
        Marc   |   110
        Simon  |   432
        ...
    '''
    query = """
            SELECT
                Customers.ContactName,
                ROUND(SUM(details.UnitPrice * details.Quantity),2) AS cumulative_amount
            FROM OrderDetails AS details
            JOIN Orders on details.OrderID = orders.OrderID
            JOIN Customers on Orders.CustomerID = customers.CustomerID
            GROUP BY Customers.ContactName
            ORDER BY cumulative_amount
         """
    return db.execute(query).fetchall()

def best_employee(db):
    '''Implement the best_employee method to determine who’s the best employee! By “best employee”, we mean the one who sells the most.
    We expect the function to return a tuple like: ('FirstName', 'LastName', 6000 (the sum of all purchase)). The order of the information is irrelevant'''
    query = """
            SELECT
                Employees.FirstName,
                Employees.LastName,
                SUM(details.UnitPrice * details.Quantity) AS cumulative_amount
            FROM OrderDetails AS details
            JOIN Orders on details.OrderID = orders.OrderID
            JOIN Employees on orders.EmployeeID = Employees.EmployeeID
            GROUP BY Employees.EmployeeID
            ORDER BY cumulative_amount DESC
            LIMIT 1
    """
    return db.execute(query).fetchone()

def orders_per_customer(db):
    '''Return a list of tuples where each tuple contains the contactName
    of the customer and the number of orders they made (contactName,
    number_of_orders). Order the list by ascending number of orders'''
    query = '''
        SELECT
            Customers.ContactName,
            COUNT(Orders.OrderID) AS order_amount
        FROM Customers
        LEFT JOIN Orders ON Customers.CustomerID = Orders.CustomerID
        GROUP BY Customers.CustomerID
        ORDER BY order_amount ASC
    '''
    return db.execute(query).fetchall()


print(orders_per_customer(db))
