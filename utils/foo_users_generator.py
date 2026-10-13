from faker import Faker
import pandas as pd
import random

# Load your existing product dataset.
# Adjust the filename and columns as needed.
products_df = pd.read_csv("./data/new_dataset/foo-amazon-products-merged-updated.csv")
# For example, assume products_df has columns:
# ['product_title', 'category', 'image_url', 'product_url', 'rating', 'review_count', 'price']

fake = Faker()


def generate_users(num_users=5):
    """Generate a DataFrame of fake users."""
    users = []
    for i in range(num_users):
        gender_rand = fake.random_element(elements=("M", "F"))
        user = {
            "user_id": i + 1,
            "name": fake.name_female() if gender_rand == "F" else fake.name_male(),
            "email": fake.email(),
            "city": fake.city(),
            "country": fake.country(),
            "gender": gender_rand,
        }
        users.append(user)
    return pd.DataFrame(users)


def generate_purchases(users_df, products_df, min_purchases=1, max_purchases=3):
    """
    Generate a DataFrame of fake purchase transactions.
    For each user, a random number of purchases is generated and each purchase is assigned
    a product from the products_df.
    """
    purchases = []
    for _, user in users_df.iterrows():
        num_purchases = random.randint(min_purchases, max_purchases)
        for j in range(num_purchases):
            # Randomly select a product from your product dataset.
            product = products_df.sample(n=1).iloc[0]
            purchase = {
                "purchase_id": f"U{user['user_id']}_{j+1}",
                "product_id": product["product_id"],
                "user_id": user["user_id"],
                "product_name": product["product_name"],
                "product_category": product["category"],
                "product_url": product["product_url"],
                "price": product["price"],
                "purchase_date": fake.date_this_year().strftime("%Y-%m-%d"),
                "quantity": random.randint(1, 5),
                "brand": product["brand"],
                "color": product["color"],
            }
            purchases.append(purchase)
    return pd.DataFrame(purchases)


if __name__ == "__main__":
    # Generate a small set of fake users.
    users_df = generate_users(num_users=500)
    # Generate fake purchase transactions referencing your products.
    purchases_df = generate_purchases(
        users_df, products_df, min_purchases=0, max_purchases=50
    )

    # Display the generated data.
    print("Users:")
    print(users_df)
    print("\nPurchases:")
    print(purchases_df)

    # Optionally, save the datasets to CSV files.
    users_df.to_csv("./data/foo_customers_upd.csv", index=False)
    purchases_df.to_csv("./data/foo_purchase_history_upd.csv", index=False)
