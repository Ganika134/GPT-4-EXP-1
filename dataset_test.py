import pandas as pd

print("========== DATASET TEST ==========")

# ---------------- PRODUCTS ----------------
print("\n===== PRODUCTS =====")

products = pd.read_csv("data/products.csv")

print(products.head())
print("\nRows:", len(products))
print("Columns:", products.columns.tolist())
print("Attributes:")
print(products.info())


# ---------------- FAQS ----------------
print("\n===== FAQS =====")

faqs = pd.read_csv(
    "data/faqs.csv",
    sep=None,
    engine="python"
)

print(faqs.head())
print("\nRows:", len(faqs))
print("Columns:", faqs.columns.tolist())
print("Attributes:")
print(faqs.info())


# ---------------- REVIEWS ----------------
print("\n===== REVIEWS =====")

reviews = pd.read_csv(
    "data/reviews.csv",
    sep=None,
    engine="python"
)

print(reviews.head())
print("\nRows:", len(reviews))
print("Columns:", reviews.columns.tolist())
print("Attributes:")
print(reviews.info())