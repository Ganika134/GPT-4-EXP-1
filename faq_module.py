import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading FAQ Module...")

# Load FAQ dataset
faqs = pd.read_csv("data/faqs.csv")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
faq_embeddings = model.encode(
    faqs["Question"].tolist(),
    convert_to_numpy=True
)

# User question
question = input("Ask Question : ")

# Generate question embedding
question_embedding = model.encode(
    [question],
    convert_to_numpy=True
)

# Calculate similarity
scores = cosine_similarity(
    question_embedding,
    faq_embeddings
)[0]

# Find best FAQ
best_index = scores.argmax()
best_score = scores[best_index]

best_question = faqs.iloc[best_index]["Question"]
best_answer = faqs.iloc[best_index]["Answer"]

print("\nBest Matching FAQ :", best_question)
print("Similarity Score :", round(best_score, 2))

# Threshold
threshold = 0.50

if best_score >= threshold:
    print("\nChatbot:")
    print(best_answer)
else:
    print("\nNo suitable FAQ found.")