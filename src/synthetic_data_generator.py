import logging
import json
import random
from pathlib import Path
from datetime import date, datetime, timedelta
import numpy as np
import polars as pl
from faker import Faker

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Generate synthetic sales data"""

    SHIP_MODES = ["First Class", "Second Class", "Standard Class"]
    SEGMENTS = ["Consumer", "Corporate", "Home Office"]
    US_CITIES = {
        "New York": {"state": "New York", "zip_prefix": "100", "region": "East"},
        "Los Angeles": {"state": "California", "zip_prefix": "900", "region": "West"},
        "Chicago": {"state": "Illinois", "zip_prefix": "606", "region": "Central"},
        "Houston": {"state": "Texas", "zip_prefix": "770", "region": "Central"},
        "Phoenix": {"state": "Arizona", "zip_prefix": "850", "region": "West"},
        "Philadelphia": {"state": "Pennsylvania", "zip_prefix": "191", "region": "East"},
        "San Antonio": {"state": "Texas", "zip_prefix": "782", "region": "Central"},
        "San Diego": {"state": "California", "zip_prefix": "921", "region": "West"},
        "Dallas": {"state": "Texas", "zip_prefix": "752", "region": "Central"},
        "San Jose": {"state": "California", "zip_prefix": "951", "region": "West"},
        "Austin": {"state": "Texas", "zip_prefix": "787", "region": "Central"},
        "Jacksonville": {"state": "Florida", "zip_prefix": "322", "region": "South"},
        "Fort Worth": {"state": "Texas", "zip_prefix": "761", "region": "Central"},
        "Columbus": {"state": "Ohio", "zip_prefix": "432", "region": "East"},
        "Charlotte": {"state": "North Carolina", "zip_prefix": "282", "region": "South"},
        "Indianapolis": {"state": "Indiana", "zip_prefix": "462", "region": "Central"},
        "Seattle": {"state": "Washington", "zip_prefix": "981", "region": "West"},
        "Denver": {"state": "Colorado", "zip_prefix": "802", "region": "West"},
        "Boston": {"state": "Massachusetts", "zip_prefix": "021", "region": "East"},
        "Nashville": {"state": "Tennessee", "zip_prefix": "372", "region": "South"},
        "Detroit": {"state": "Michigan", "zip_prefix": "482", "region": "Central"},
        "Portland": {"state": "Oregon", "zip_prefix": "972", "region": "West"},
        "Las Vegas": {"state": "Nevada", "zip_prefix": "891", "region": "West"},
        "Miami": {"state": "Florida", "zip_prefix": "331", "region": "South"},
        "Atlanta": {"state": "Georgia", "zip_prefix": "303", "region": "South"},
    }

    def __init__(self, input_dir: str | Path = "data/input", seed: int = 42, locale: str = "en_US") -> None:
        """Initialize the synthetic data generator"""
        self.input_dir = Path(input_dir)
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.fake = Faker(locale)
        Faker.seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        logger.info("Initialized SyntheticDataGenerator with seed=%s", seed)

    def generate_customer_name(self) -> str:
        """Generate a customer name"""
        return self.fake.name()

    def generate_customer_id(self, customer_name: str) -> str:
        """Generate a customer identifier (e.g. CG-12456)."""
        parts = customer_name.split()
        if len(parts) < 2:
            initials = (parts[0][0] if parts else "X") + "X"
        else:
            initials = parts[0][0] + parts[1][0]
        number = random.randint(10000, 99999)
        return f"{initials.upper()}-{number}"

    def generate_order_date( self, start_date: date, end_date: date ) -> date:
        """Generate order date"""
        order_date = self.fake.date_between(start_date=start_date, end_date=end_date)
        return order_date
    
    def generate_ship_date(self, order_date: date, min_ship_days: int = 1 , max_ship_days: int = 7)-> date:
        """Generate ship_date"""
        ship_date =  self.fake.date_between(
            start_date=order_date + timedelta(days=min_ship_days),
            end_date=order_date + timedelta(days=max_ship_days) )
        return ship_date
    
    def generate_ship_mode(self, order_date: date, ship_date: date, ) -> str:
        if order_date == ship_date:
            return "Same Day"
        else:
            return random.choice(self.SHIP_MODES)
    
    def generate_location_data(self) -> dict[str, str]:
        """Generate city, state, postal code and region"""
        city = random.choice(list(self.US_CITIES))
        city_info = self.US_CITIES[city]
        postal_code = f"{city_info['zip_prefix']}{random.randint(10, 99)}"
        return {
            "City": city,
            "State": city_info["state"],
            "Postal Code": postal_code,
            "Region": city_info["region"],
        }

    def generate_product_details(self) -> dict[str, str]:
        """Generate category and sub_category"""
        catalog_path = self.input_dir / "product_catalog.json"
        with catalog_path.open("r") as f:
            product_catalog = json.load(f)
        category = random.choice(list(product_catalog.keys()))
        sub_category = random.choice(list(product_catalog[category].keys()))
        product = random.choice(list(product_catalog[category][sub_category]))
        return category, sub_category, product

    def generate_order_id(self, order_date: datetime) -> str:
        """Generate an order identifier (e.g. US-2016-118983)"""
        return f"US-{order_date.year}-{random.randint(100000, 999999)}"
    
    def generate_quantity(self, category: str = None) -> int:
        """Generate quantity"""
        if category == "Furniture":
            return random.randint(1, 5)
        elif category == "Technology":
            return random.randint(1,20)
        else:
            return random.randint(1,100)
    
    def generate_discount(self, quantity: int) -> float:
        """Generate discount percentages"""
        if quantity >= 100:
            return  0.15
        elif quantity >= 20:
            return 0.10
        elif quantity >=5:
            return 0.05
        else:
            return 0

    def generate_product_id(self, category: str, subcategory: str) -> str:
        """Generate a product identifier based on category and sub-category"""
        cat_abbr = category[:3].upper()
        sub_abbr = subcategory[:2].upper()
        return f"{cat_abbr}-{sub_abbr}-100{random.randint(10000, 99999)}"
    
    def generate_synthetic_data(
        self,
        num_rows: int,
        start_date: date,
        end_date: date,
    ) -> pl.DataFrame:
        """Generate synthetic sales records"""

        logger.info("Generating %s synthetic rows...", num_rows)

        synthetic_rows = []

        for row_id in range(1, num_rows + 1):
            customer_name = self.generate_customer_name()
            customer_id = self.generate_customer_id(customer_name)
            order_date = self.generate_order_date(
                start_date=start_date, end_date=end_date)
            ship_date = self.generate_ship_date(order_date=order_date)
            order_id = self.generate_order_id(order_date=order_date)
            ship_mode = self.generate_ship_mode(order_date=order_date, ship_date=ship_date)
            location = self.generate_location_data()
            category, sub_category, product = self.generate_product_details()
            product_price = product['price']
            product_name = product['name']
            product_id = self.generate_product_id(category, sub_category)
            quantity = self.generate_quantity(category= category)
            discount = self.generate_discount(quantity=quantity)

            row = {
                "row_id": row_id,
                "order_id": order_id,
                "order_date": order_date,
                "ship_date": ship_date,
                "ship_mode": ship_mode,
                "customer_id": customer_id,
                "customer_name": customer_name,
                "purchasing_frequency": random.choice([
                'Monthly', 'Quarterly', 'Bi-Annually']),
                "segment": random.choice(self.SEGMENTS),
                "country": "United States",
                "city": location["City"],
                "state": location["State"],
                "postal_code": location["Postal Code"],
                "region": location["Region"],
                "category": category,
                "sub_category": sub_category,
                "product_id": product_id,
                "product_name": product_name,
                "price": product_price,
                "quantity": quantity,
                "discount": discount
            }
            synthetic_rows.append(row)

        synthetic_df = pl.DataFrame(synthetic_rows)
        logger.info("Successfully generated %s synthetic rows", len(synthetic_df))
        return synthetic_df
