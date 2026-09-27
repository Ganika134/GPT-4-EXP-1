import pandas as pd
import re

print("===== FIXING REVIEWS DATASET =====")

# Load products
products = pd.read_csv("data/products.csv")

# Clean product information
products["ProductID"] = pd.to_numeric(
    products["ProductID"], errors="coerce"
)

products["ProductName"] = (
    products["ProductName"]
    .fillna("")
    .astype(str)
    .str.strip()
)

# ProductID -> ProductName
product_map = {
    str(int(row["ProductID"])): row["ProductName"]
    for _, row in products.dropna(subset=["ProductID"]).iterrows()
}

# Read original file
with open("data/reviews.csv", "r", encoding="utf-8") as f:
    lines = f.readlines()

fixed_reviews = []
failed = 0

for line_number, line in enumerate(lines[1:], start=2):

    line = line.strip().strip('"')

    if not line:
        continue

    # Get ReviewID and ProductID from the beginning
    first = re.match(r"^(\d+)\s+(\d+)\s+(.*)$", line)

    if not first:
        failed += 1
        continue

    review_id = first.group(1)
    product_id = first.group(2)
    remaining = first.group(3).strip()

    # Rating = final number
    rating_match = re.search(r"\s+(\d+)\s*$", remaining)

    if not rating_match:
        failed += 1
        continue

    rating = rating_match.group(1)

    # Remove rating
    text = remaining[:rating_match.start()].strip()

    # Get product name
    product_name = product_map.get(product_id, "")

    if not product_name:
        failed += 1
        continue

    # Find product name
    position = text.find(product_name)

    if position == -1:
        failed += 1
        continue

    # Everything after product name is review
    review = text[position + len(product_name):].strip()

    if not review:
        failed += 1
        continue

    fixed_reviews.append([
        int(review_id),
        int(product_id),
        product_name,
        review,
        int(rating)
    ])

# Create dataframe
fixed = pd.DataFrame(
    fixed_reviews,
    columns=[
        "ReviewID",
        "ProductID",
        "ProductName",
        "Review",
        "Rating"
    ]
)

# Save
fixed.to_csv("data/reviews_fixed.csv", index=False)

print()
print("Original rows :", len(lines) - 1)
print("Fixed rows    :", len(fixed))
print("Failed rows   :", failed)

print()
print("Columns:")
print(fixed.columns.tolist())

print()
print("First 10 rows:")
print(fixed.head(10).to_string(index=False))

print()
print("Saved:")
print("data/reviews_fixed.csv")