import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading Product QA Module...")

# Load products
products = pd.read_csv("data/products.csv")

print("Products loaded:", len(products))

# Handle missing values
for column in ["ProductName", "Category", "Brand", "Description"]:
    products[column] = products[column].fillna("").astype(str)

# Create searchable text
products["text"] = (
    products["ProductName"] + " " +
    products["Category"] + " " +
    products["Brand"] + " " +
    products["Description"]
)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
vectors = vectorizer.fit_transform(products["text"])

# Ask product question
query = input("\nAsk Product Question : ")

query_vector = vectorizer.transform([query])

# Find matching products
scores = cosine_similarity(query_vector, vectors)[0]

top_indices = scores.argsort()[-3:][::-1]

print("\n===== PRODUCT SEARCH RESULTS =====")

for rank, index in enumerate(top_indices, start=1):

    product = products.iloc[index]

    print(f"\n{rank}. {product['ProductName']}")
    print("Product ID :", product["ProductID"])
    print("Brand :", product["Brand"])
    print("Category :", product["Category"])
    print("Price : ₹", product["Price"])
    print("Description :", product["Description"])
    print("Similarity :", round(scores[index], 3))