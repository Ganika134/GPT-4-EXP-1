import pandas as pd
from transformers import pipeline

# ==============================
# LOAD FIXED REVIEWS DATASET
# ==============================

reviews = pd.read_csv("data/reviews_fixed.csv")

# Remove unwanted spaces from column names
reviews.columns = reviews.columns.str.strip()

# Clean ProductName and Review columns
reviews["ProductName"] = reviews["ProductName"].astype(str).str.strip()
reviews["Review"] = reviews["Review"].astype(str).str.strip()

# ==============================
# DISPLAY AVAILABLE PRODUCTS
# ==============================

print("\n===== REVIEW INTELLIGENCE =====")

product = input("Enter Product : ").strip()

# ==============================
# FIND PRODUCT REVIEWS
# ==============================

product_reviews = reviews[
    reviews["ProductName"].str.lower() == product.lower()
]

if product_reviews.empty:
    print("\nNo reviews found for:", product)
    print("\nAvailable products:")

    for name in reviews["ProductName"].drop_duplicates():
        print("-", name)

    exit()

# ==============================
# LOAD SENTIMENT MODEL
# ==============================

print("\nLoading sentiment model...")

sentiment_model = pipeline("sentiment-analysis")

# ==============================
# SENTIMENT COUNTERS
# ==============================

positive = 0
neutral = 0
negative = 0

print("\n===== DETAILED REVIEW =====")

# ==============================
# ANALYZE EACH REVIEW
# ==============================

for review in product_reviews["Review"]:

    result = sentiment_model(review)[0]

    label = result["label"]
    score = result["score"]

    # Convert model output into 3 categories
    if label == "POSITIVE":

        positive += 1
        display_label = "Positive"

    else:

        # Simple rule for neutral sentiment
        if score < 0.75:
            neutral += 1
            display_label = "Neutral"
        else:
            negative += 1
            display_label = "Negative"

    print("------------------")
    print(review)
    print(display_label)
    print(round(score, 3))


# ==============================
# REVIEW SUMMARY
# ==============================

print("\n===== REVIEW SUMMARY =====")

print("Positive :", positive)
print("Neutral  :", neutral)
print("Negative :", negative)

print("\n===== TOTAL REVIEWS =====")
print("Total :", len(product_reviews))