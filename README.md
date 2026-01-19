# Information Retrieval Methods

## A. Sparse (Keyword-Based) Retrieval

### 1. BM25
BM25 is a probabilistic term-weighting retrieval model that scores documents using term frequency, inverse document frequency, and document length normalization.

**Support Points:**
- Remains a strong, robust baseline across domains (BEIR benchmark).
- Competitive even against modern neural retrievers.
- Simple, interpretable, and fast.

**References:**
- [The Probabilistic Relevance Framework: BM25 and Beyond (Robertson & Zaragoza, 2009)](https://staff.city.ac.uk/~sb317/)
- [BEIR Benchmark — BM25 as a robust baseline](https://arxiv.org/abs/2104.08663)

---

## B. Dense (Embedding-Based) Retrieval

### 2. Dense Vector Search (FAISS, Milvus, Qdrant, Weaviate)
Dense retrieval encodes queries and documents into vectors using neural encoders and retrieves via approximate nearest-neighbor search.

**Support Points:**
- Outperforms BM25 by 9–19% on top-20 recall (DPR).
- Efficient embeddings for similarity search (Sentence-BERT).
- Zero-shot performance superior to BM25 (E5).

**References:**
- [DPR: Dense Passage Retrieval](https://arxiv.org/abs/2004.04906)
- [Sentence-BERT](https://arxiv.org/abs/1908.10084)
- [E5 Embeddings](https://arxiv.org/abs/2204.02984)

### 3. Domain-Adapted Embeddings
Dense models fine-tuned on domain-specific pairs significantly improve contextual retrieval accuracy.

**Support Points:**
- Fine-tuning boosts in-domain performance.
- Dense retrievers often underperform out-of-domain unless adapted.
- Contriever improves zero-shot but benefits from targeted data.

**References:**
- [BEIR: Dense models OOD limitations](https://arxiv.org/abs/2104.08663)
- [Contriever](https://arxiv.org/abs/2112.04426)

---

## C. Hybrid Retrieval (Dense + Sparse)

### 4. Hybrid Dense–Sparse Retrieval (BM25 + Dense)
Combines BM25’s lexical strength with dense retrieval’s semantic matching for higher recall and robustness.

**Support Points:**
- Outperforms either method alone (BEIR).
- Robust to domain shifts and heterogeneous corpora.

**References:**
- [BEIR: Hybrid methods](https://arxiv.org/abs/2104.08663)

### 5. Reciprocal Rank Fusion (RRF)
RRF merges multiple retrieval lists and consistently outperforms individual retrievers.

**Support Points:**
- Simple, effective rank-fusion algorithm.
- Improves on the best of combined results.
- Commonly paired with hybrid and multi-query retrieval.

**References:**
- [RRF Paper](https://cormack.uwaterloo.ca/cormack/papers/Reciprocal_Rank_Fusion.pdf)

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
