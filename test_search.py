from services.embedding_service import generate_embedding
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "Artificial intelligence is transforming companies.",
    "Football is the most popular sport in the world.",
    "Machine learning is a branch of AI."
]

query = "What is AI?"

query_embedding = generate_embedding(query)

scores = []

for doc in documents:
    doc_embedding = generate_embedding(doc)

    similarity = cosine_similarity(
        [query_embedding],
        [doc_embedding]
    )[0][0]

    scores.append((doc, similarity))

scores.sort(key=lambda x: x[1], reverse=True)

for doc, score in scores:
    print(f"{score:.4f} -> {doc}")