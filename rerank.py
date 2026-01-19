from sentence_transformers import SentenceTransformer

model_name = 'sentence-transformers/paraphrase-xlm-r-multilingual-v1'
model = SentenceTransformer(model_name)
documents = [
    "This is a list which containing sample documents.",
    "Keywords are important for keyword-based search.",
    "Document analysis involves extracting keywords.",
    "Keyword-based search relies on sparse embeddings.",
    "Understanding document structure aids in keyword extraction.",
    "Efficient keyword extraction enhances search accuracy.",
    "Semantic similarity improves document retrieval performance.",
    "Machine learning algorithms can optimize keyword extraction methods."
]

document_embeddings = model.encode(documents)
for i, embedding in enumerate(document_embeddings):
    print(f"Document {i+1} embedding: {embedding}")

query = "Natural language processing techniques enhance keyword extraction efficiency."
query_embedding = model.encode(query)


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity(np.array([query_embedding]), document_embeddings)
most_similar_index = np.argmax(similarities)
most_similar_document = documents[most_similar_index]
similarity_score = similarities[0][most_similar_index]
sorted_indices = np.argsort(similarities[0])[::-1]
ranked_documents = [(documents[i], similarities[0][i]) for i in sorted_indices]
print("Ranked Documents:")
for rank, (document, similarity) in enumerate(ranked_documents, start=1):
    print(f"Rank {rank}: Document - '{document}', Similarity Score - {similarity}")

print("Top 4 Documents:")
print(f"Query: {query}")
for rank, (document, similarity) in enumerate(ranked_documents[:4], start=1):
    print(f"Rank {rank}: Document - '{document}', Similarity Score - {similarity}")

from rank_bm25 import BM25Okapi

top_4_documents = [doc[0] for doc in ranked_documents[:4]]
tokenized_top_4_documents = [doc.split() for doc in top_4_documents]
tokenized_query = query.split()
bm25=BM25Okapi(tokenized_top_4_documents)
bm25_scores = bm25.get_scores(tokenized_query)
sorted_indices2 = np.argsort(bm25_scores)[::-1]
reranked_documents = [(top_4_documents[i], bm25_scores[i]) for i in sorted_indices2]
print("Rerank of top 4 Documents:")
for rank, (document, similarity) in enumerate(reranked_documents, start=1):
    print(f"Rank {rank}: Document - '{document}', Similarity Score - {similarity}")

from sentence_transformers import CrossEncoder
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

pairs = []
for doc in top_4_documents:
    pairs.append([query, doc])

scores = cross_encoder.predict(pairs)
scored_docs = zip(scores, top_4_documents)
reranked_document_cross_encoder = sorted(scored_docs, reverse=True)

cohere_api_key = "KgvZmcIdsoMLUhmCvjwqAyAYiIVGQ9zOwlPEv58e"
import cohere
co = cohere.Client(cohere_api_key)

response = co.rerank(
    model="rerank-english-v3.0",
    query="Natural language processing techniques enhance keyword extraction efficiency.",
    documents=top_4_documents,
    return_documents=True
)
print(response)

for i in range(4):
  print(f'text: {response.results[i].document.text} score: {response.results[i].relevance_score}')



