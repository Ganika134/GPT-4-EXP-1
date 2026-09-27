import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading Recommendation Module...")

# Load products
products = pd.read_csv("data/products.csv")

print("Products loaded:", len(products))

# Replace missing values
for column in ["ProductName", "Category", "Brand", "Description"]:
    products[column] = products[column].fillna("").astype(str)

# Combine product information
products["text"] = (
    products["ProductName"] + " " +
    products["Category"] + " " +
    products["Brand"] + " " +
    products["Description"]
)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
product_vectors = vectorizer.fit_transform(products["text"])

# User query
query = input("\nSearch Product : ")

query_vector = vectorizer.transform([query])

# Calculate similarity
scores = cosine_similarity(query_vector, product_vectors)[0]

# Get top 3
top_indices = scores.argsort()[-3:][::-1]

print("\n===== TOP 3 RECOMMENDED PRODUCTS =====")

for rank, index in enumerate(top_indices, start=1):

    product = products.iloc[index]

    print(f"\n{rank}. {product['ProductName']}")
    print("Brand :", product["Brand"])
    print("Category :", product["Category"])
    print("Price : ₹", product["Price"])
    print("Similarity :", round(scores[index], 3))