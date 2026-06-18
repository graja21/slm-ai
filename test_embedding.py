from services.embedding_service import generate_embedding

text = "Artificial Intelligence is changing the world."

embedding = generate_embedding(text)

print("Vector size:", len(embedding))
print("First 5 values:", embedding[:5])