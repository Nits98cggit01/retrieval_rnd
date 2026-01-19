# Information Retrieval Methods

## A. Sparse (Keyword-Based) Retrieval

### 1. BM25
BM25 is a probabilistic term-weighting retrieval model that scores documents using term frequency, inverse document frequency, and document length normalization.

**Support Points:**
- Remains a strong, robust baseline across domains (BEIR benchmark).
- Competitive even against modern neural retrievers.
- Simple, interpretable, and fast.

**References:**
- [BM25 - Overview]([https://staff.city.ac.uk/~sb317/](https://www.geeksforgeeks.org/nlp/what-is-bm25-best-matching-25-algorithm/))
- [BM25 Explained: A Better Ranking Algorithm than TF-IDF]([https://arxiv.org/abs/2104.08663](https://vishwasg.dev/blog/2025/01/20/bm25-explained-a-better-ranking-algorithm-than-tf-idf/))

---

## B. Dense (Embedding-Based) Retrieval

### 2. Dense Vector Search (FAISS, Milvus, Qdrant, Weaviate)
Dense retrieval encodes queries and documents into vectors using neural encoders and retrieves via approximate nearest-neighbor search.

**Support Points:**
- Outperforms BM25 by 9–19% on top-20 recall (DPR).
- Efficient embeddings for similarity search (Sentence-BERT).
- Zero-shot performance superior to BM25 (E5).

**References:**
- [FAISS]([https://arxiv.org/abs/2004.04906](https://faiss.ai/))
- [Sentence-BERT]([https://arxiv.org/abs/1908.10084](https://www.sbert.net/))
- [Apache Solr Reference Guide]([https://arxiv.org/abs/2204.02984](https://solr.apache.org/guide/solr/latest/query-guide/dense-vector-search.html))

### 3. Domain-Adapted Embeddings
Dense models fine-tuned on domain-specific pairs significantly improve contextual retrieval accuracy.

**Support Points:**
- Fine-tuning boosts in-domain performance.
- Dense retrievers often underperform out-of-domain unless adapted.
- Contriever improves zero-shot but benefits from targeted data.

**References:**
- [Domain Adapted Word Embeddings for Improved Sentiment Classification]([https://arxiv.org/abs/2104.08663](https://aclanthology.org/W18-3407/))
- [Domain adaptation for embedding models]([https://arxiv.org/abs/2112.04426](https://zilliz.com/ai-faq/what-is-domain-adaptation-for-embedding-models))

---

## C. Hybrid Retrieval (Dense + Sparse)

### 4. Hybrid Dense–Sparse Retrieval (BM25 + Dense)
Combines BM25’s lexical strength with dense retrieval’s semantic matching for higher recall and robustness.

**Support Points:**
- Outperforms either method alone (BEIR).
- Robust to domain shifts and heterogeneous corpora.

**References:**
- [A Complete Guide to Implementing Hybrid RAG]([https://arxiv.org/abs/2104.08663](https://medium.com/aingineer/a-complete-guide-to-implementing-hybrid-rag-86c0febba474))

### 5. Reciprocal Rank Fusion (RRF)
RRF merges multiple retrieval lists and consistently outperforms individual retrievers.

**Support Points:**
- Simple, effective rank-fusion algorithm.
- Improves on the best of combined results.
- Commonly paired with hybrid and multi-query retrieval.

**References:**
- [RRF Paper](https://cormack.uwaterloo.ca/cormack/papers/Reciprocal_Rank_Fusion.pdf)
- [Relevance scoring in hybrid search using Reciprocal Rank Fusion (RRF)](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking)

---

## D. Advanced Retrieval Optimization

### 6. Cross-Encoder / ColBERT Reranking
MonoT5/T5 rerankers use deep cross-attention for state-of-the-art precision; ColBERT uses late interaction for speed.

**Support Points:**
- SOTA precision by rescoring top-k candidates.
- ColBERT achieves near cross-encoder accuracy with higher speed.

**References:**
- [MonoT5 / T5 rerankers](https://aclanthology.org/2020.findings-emnlp.372/)
- [ColBERT / ColBERTv2](https://arxiv.org/abs/2004.12832), [ColBERTv2](https://aclanthology.org/2022.emnlp-main.556/)

### 7. Multi-Query Retrieval
Submitting multiple reformulated queries increases recall and robustness.

**Support Points:**
- Boosts recall for ambiguous or underspecified prompts.
- Often paired with RRF for stronger outputs.

**References:**
- [RRF Paper](https://cormack.uwaterloo.ca/cormack/papers/Reciprocal_Rank_Fusion.pdf)

### 8. Query Rewriting / Acronym Expansion / Synonym Expansion
Techniques to mitigate vocabulary mismatch and improve recall.

**Support Points:**
- Pseudo relevance feedback (RM3) expands queries using top documents.
- Controlled expansion avoids topic drift.

**References:**
- [RM3](https://nlp.stanford.edu/IR-book/html/htmledition/pseudo-relevance-feedback-1.html)
- [Query expansion studies](https://onlinelibrary.wiley.com/doi/abs/10.1002/asi.10102)

### 9. HyDE (Hypothetical Document Embeddings)
HyDE uses LLMs to generate hypothetical documents for embedding and retrieval.

**Support Points:**
- Large zero-shot gains over unsupervised dense baselines.
- Matches fine-tuned retrievers in many tasks.

**References:**
- [HyDE](https://aclanthology.org/2023.acl-long.801/)

---

## E. Structured & Graph-Aware Retrieval

### 10. Knowledge Graph Retrieval / GraphRAG
Uses entities, relationships, and graph traversal for contextually relevant knowledge retrieval.

**Support Points:**
- Enables structured reasoning for long-context or fact-heavy domains.
- Strong for entity-centric tasks.

---

## F. Metadata-Driven Retrieval

### 11. Metadata Filters
Filtering by attributes improves precision by restricting search scope.

**Support Points:**
- Drastically improves precision in enterprise systems.
- BM25F incorporates document fields for enhanced retrieval.

**References:**
- [BM25F field weighting](https://staff.city.ac.uk/~sb317/)

---

## Ranking: Best to Worst (with Explanations & Citations)

1. **Hybrid Dense–Sparse Retrieval + RRF (BEST OVERALL)**
   - Best balance of recall, precision, OOD robustness, and stability.
   - BM25 robust baseline + dense semantics + RRF combination stability.
   - Citations: [arxiv.org](https://arxiv.org/abs/2104.08663), [cormack.uwaterloo.ca](https://cormack.uwaterloo.ca/cormack/papers/Reciprocal_Rank_Fusion.pdf)

2. **Cross-Encoder / ColBERT Reranking (WITH Hybrid Retrieval)**
   - SOTA precision by rescoring top-k candidates.
   - Citations: [aclanthology.org](https://aclanthology.org/2020.findings-emnlp.372/), [arxiv.org](https://arxiv.org/abs/2004.12832)

3. **Query Rewriting + Acronym / Synonym Expansion**
   - Improves lexical matching, especially in expert or acronym-heavy domains.
   - Citations: [nlp.stanford.edu](https://nlp.stanford.edu/IR-book/html/htmledition/pseudo-relevance-feedback-1.html), [onlinelibrary.wiley.com](https://onlinelibrary.wiley.com/doi/abs/10.1002/asi.10102)

4. **HyDE (Hypothetical Document Embedding)**
   - Large zero-shot boost where no training data exists.
   - Citations: [aclanthology.org](https://aclanthology.org/2023.acl-long.801/)

5. **Knowledge Graph Retrieval / GraphRAG**
   - Strong for reasoning and entity-centric domains, but requires structured data.
   - (No direct citation in provided results)

6. **Domain-Adapted Embeddings**
   - Powerful when domain-specific parallel data exists; weaker in transfer.
   - Citations: [arxiv.org](https://arxiv.org/abs/2104.08663), [arxiv.org](https://arxiv.org/abs/2112.04426)

7. **Metadata-Driven Retrieval**
   - Great for precision, filtering noise, enforcing compliance; limited in semantic recall.
   - Citations: [staff.city.ac.uk](https://staff.city.ac.uk/~sb317/)

8. **Dense-Only Retrieval**
   - Fast and semantic, but brittle OOD unless hybridized or reranked.
   - Citations: [arxiv.org](https://arxiv.org/abs/2104.08663)

9. **Sparse-Only Retrieval (BM25 only)**
   - Strong baseline but misses paraphrasing and semantic intent.
   - Citations: [arxiv.org](https://arxiv.org/abs/2104.08663)

10. **Vector + Naive Reranking (Similarity Only)**
    - Simple cosine-similarity reranking lacks true semantic composition or query-document interaction.
    - Performs worst among modern options.


---

## Roadmap for Top 3 Retrieval Methods

### 1. Hybrid Dense–Sparse Retrieval + RRF

**Step 1:** Implement BM25 for sparse retrieval as a baseline.  
**Step 2:** Integrate a dense retriever (e.g., Sentence-BERT, DPR, or E5) for semantic search.  
**Step 3:** Combine results from both retrievers using Reciprocal Rank Fusion (RRF) to merge rankings.  
**Step 4:** Evaluate recall, precision, and robustness on in-domain and out-of-domain datasets.  
**Step 5:** Tune fusion weights and thresholds for optimal performance.

---

### 2. Cross-Encoder / ColBERT Reranking (WITH Hybrid Retrieval)

**Step 1:** Use the hybrid retrieval pipeline (BM25 + dense retriever) to generate a candidate set (top-k).  
**Step 2:** Apply a cross-encoder (e.g., MonoT5, T5) or ColBERT reranker to rescore the candidate set using deep interaction.  
**Step 3:** Select the top results based on reranker scores.  
**Step 4:** Benchmark precision and latency; optimize reranker batch size and inference speed as needed.

---

### 3. Query Rewriting + Acronym / Synonym Expansion

**Step 1:** Analyze queries for possible ambiguities, acronyms, or synonyms.  
**Step 2:** Apply query rewriting techniques (manual rules, synonym dictionaries, or LLM-based rewriting).  
**Step 3:** Expand queries using pseudo relevance feedback (e.g., RM3) or external knowledge bases.  
**Step 4:** Submit multiple reformulated queries to the retrieval system and aggregate results (optionally with RRF).  
**Step 5:** Monitor for topic drift and tune expansion strategies for domain relevance.

---

## Comparison Table

| Method                                      | Recall | Precision | OOD Robustness | Speed   | Implementation Complexity | Data Requirements         | Strengths                                   | Weaknesses                                  |
|----------------------------------------------|--------|-----------|----------------|---------|--------------------------|---------------------------|---------------------------------------------|---------------------------------------------|
| Hybrid Dense–Sparse Retrieval + RRF          | High   | High      | High           | Medium  | Medium                   | Moderate (BM25 + dense)   | Best overall, robust, stable                | Needs fusion logic, more infra              |
| Cross-Encoder / ColBERT Reranking (Hybrid)   | High   | SOTA      | High           | Low-Med | High                     | Large for reranking       | SOTA precision, deep interaction            | Slow, compute-intensive                     |
| Query Rewriting + Acronym/Synonym Expansion  | High   | Medium    | Medium         | Medium  | Medium                   | Synonym/acronym resources | Handles vocab mismatch, boosts recall       | Risk of topic drift, tuning needed           |
| HyDE (Hypothetical Document Embedding)       | High   | Medium    | High           | Medium  | Medium                   | LLM access                | Zero-shot, no training data needed          | LLM cost, quality varies                    |
| Knowledge Graph Retrieval / GraphRAG         | Medium | High      | Medium         | Low     | High                     | Structured data/graphs    | Structured reasoning, entity-centric        | Needs graph infra, limited applicability    |
| Domain-Adapted Embeddings                    | High   | Medium    | Low-Med        | High    | High                     | Domain pairs for tuning   | Best in-domain, contextual                  | Poor OOD, needs labeled data                |
| Metadata-Driven Retrieval                    | Low    | High      | High           | High    | Low                      | Metadata-rich docs        | Precision, compliance, filtering            | Limited semantic recall                     |
| Dense-Only Retrieval                         | Medium | Medium    | Low            | High    | Medium                   | Dense encoder             | Fast, semantic                             | Brittle OOD, misses lexical cues            |
| Sparse-Only Retrieval (BM25 only)            | Medium | Medium    | High           | High    | Low                      | None                      | Simple, robust, interpretable               | Misses semantics, paraphrasing              |
| Vector + Naive Reranking (Similarity Only)   | Low    | Low       | Low            | High    | Low                      | Dense encoder             | Simple, fast                               | Weak semantic matching, poor accuracy       |

---

*OOD: Out-of-Domain; SOTA: State-of-the-Art*
