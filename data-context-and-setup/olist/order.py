import pandas as pd
import numpy as np
from olist.utils import haversine_distance
from olist.data import Olist


class Order:
    '''
    DataFrames containing all orders as index,
    and various properties of these orders as columns
    '''
    def __init__(self):
        # Assign an attribute ".data" to all new instances of Order
        self.data = Olist().get_data()

    def get_wait_time(self, is_delivered=True):
        """
        Returns a DataFrame with:
        [order_id, wait_time, expected_wait_time, delay_vs_expected, order_status]
        and filters out non-delivered orders unless specified
        """
        # Hint: Within this instance method, you have access to the instance of the class Order in the variable self, as well as all its attributes
        orders = self.data['orders'].copy()

        # filter delivered orders only
        if is_delivered:
            orders = orders.query("order_status=='delivered'").copy()

        # handle datetime
        orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
        orders['order_estimated_delivery_date'] = pd.to_datetime(orders['order_estimated_delivery_date'])
        orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])

        # compute delay vs expected
        orders.loc[:, 'delay_vs_expected'] = \
            (orders['order_delivered_customer_date'] - orders['order_estimated_delivery_date']) / np.timedelta64(24, 'h')

        orders.loc[:,'delay_vs_expected'] = orders['delay_vs_expected'].apply(lambda x: x if x > 0 else 0)

        # compute wait_time
        orders.loc[:, 'wait_time'] = \
            (orders['order_delivered_customer_date'] - orders['order_purchase_timestamp']) / np.timedelta64(24, 'h')

        # compute expected wait time
        orders.loc[:, 'expected_wait_time'] = \
            (orders['order_estimated_delivery_date'] - orders['order_purchase_timestamp']) / np.timedelta64(24, 'h')

        return orders[[
            'order_id', 'wait_time', 'expected_wait_time', 'delay_vs_expected', 'order_status'
            ]]


    def get_review_score(self):
        """
        Returns a DataFrame with:
        order_id, dim_is_five_star, dim_is_one_star, review_score
        """
        reviews = self.data['order_reviews'].copy()

        dim_is_one_star = lambda x: int(x==1)
        dim_is_five_star = lambda x: int(x==5)

        reviews['dim_is_five_star'] = reviews['review_score'].apply(dim_is_five_star)
        reviews['dim_is_one_star'] = reviews['review_score'].apply(dim_is_one_star)

        return reviews[[
            "order_id", "dim_is_five_star", "dim_is_one_star", "review_score"
            ]]

    def get_number_products(self):
        """
        Returns a DataFrame with:
        order_id, number_of_products
        """
        order_items = self.data['order_items'].copy()

        products = order_items.groupby('order_id', as_index=False).agg({'order_item_id': 'count'})
        products.columns = ['order_id', 'number_of_products']

        return products

    def get_number_sellers(self):
        """
        Returns a DataFrame with:
        order_id, number_of_sellers
        """
        sellers = self.data['order_items'].copy()

        sellers = sellers.groupby('order_id')['seller_id'].nunique().reset_index()
        sellers.columns = ['order_id', 'number_of_sellers']

        return sellers

    def get_price_and_freight(self):
        """
        Returns a DataFrame with:
        order_id, price, freight_value
        """
        price_freight = self.data['order_items'].copy()

        price_freight = price_freight.groupby('order_id', as_index=False).agg({'price': 'sum', 'freight_value': 'sum'})

        return price_freight[['order_id', 'price', 'freight_value']]

    # Optional
    def get_distance_seller_customer(self):
        """
        Returns a DataFrame with:
        order_id, distance_seller_customer
        """
        data = self.data
        orders = data['orders']
        order_items = data['order_items']
        sellers = data['sellers']
        customers = data['customers']

        # since one zip code can map to multiple lat/lon, take the first one
        geo = data['geolocation']
        geo = geo.groupby('geolocation_zip_code_prefix', as_index=False).first()

        # merge geolocation for sellers
        sellers_mask_columns = ['seller_id', 'seller_zip_code_prefix', 'geolocation_lat', 'geolocation_lng']

        sellers_geo = sellers.merge(
                            geo,
                            how='left',
                            left_on='seller_zip_code_prefix',
                            right_on='geolocation_zip_code_prefix')[sellers_mask_columns]

        # merge geolocation for customers
        customers_mask_columns = ['customer_id', 'customer_zip_code_prefix', 'geolocation_lat', 'geolocation_lng']

        customers_geo = customers.merge(
                            geo,
                            how='left',
                            left_on='customer_zip_code_prefix',
                            right_on='geolocation_zip_code_prefix')[customers_mask_columns]

        # match customers with sellers in one table
        customer_sellers = customers.merge(orders, on='customer_id')\
        .merge(order_items, on='order_id')\
        .merge(sellers, on='seller_id')\
        [['order_id', 'customer_id','customer_zip_code_prefix', 'seller_id', 'seller_zip_code_prefix']]

        # add the geoloc
        matching_geo = customer_sellers.merge(sellers_geo, on='seller_id')\
                    .merge(customers_geo, on='customer_id', suffixes=('_seller', '_customer'))

        # drop na()
        matching_geo = matching_geo.dropna()

        matching_geo.loc[:, 'distance_seller_customer'] = \
            matching_geo.apply(lambda row: haversine_distance(row['geolocation_lat_seller'],
                                                              row['geolocation_lng_seller'],
                                                              row['geolocation_lat_customer'],
                                                              row['geolocation_lng_customer']),
                                                              axis=1)

        # Since an order can have multiple sellers,
        # return the average of the distance per order
        order_distance = matching_geo.groupby('order_id', as_index=False).agg({'distance_seller_customer': 'mean'})

        return order_distance

    def get_training_data(self,
                          is_delivered=True,
                          with_distance_seller_customer=False):
        """
        Returns a clean DataFrame (without NaN), with the all following columns:
        ['order_id', 'wait_time', 'expected_wait_time', 'delay_vs_expected',
        'order_status', 'dim_is_five_star', 'dim_is_one_star', 'review_score',
        'number_of_products', 'number_of_sellers', 'price', 'freight_value',
        'distance_seller_customer']
        """
        # Hint: make sure to re-use your instance methods defined above
        training_set = self.get_wait_time(is_delivered)\
            .merge(
                self.get_review_score(), on='order_id')\
            .merge(
                self.get_number_products(), on='order_id')\
            .merge(
                self.get_number_sellers(), on='order_id')\
            .merge(
                self.get_price_and_freight(), on='order_id')\

        if with_distance_seller_customer:
            training_set = training_set.merge(
                self.get_distance_seller_customer(), on='order_id')

        return training_set.dropna()
