# FoodBot — Vietnamese Recipe Recommendation & Chatbot

[▶️ Demo Video](https://youtu.be/n5z3Xe-FaHc)

## Overview
**FoodBot** is a Vietnamese recipe recommendation system that blends Natural Language Processing (NLP) with classic information-retrieval techniques to suggest dishes from free-text user queries. The project provides **five complementary recommenders** and an **interactive chatbot** that understands Vietnamese, extracts ingredients and intent, and returns suitable recipes. A Flask web UI lets you compare model outputs side-by-side, while a background data routine can enrich the recipe corpus via the Spoonacular API. :contentReference[oaicite:0]{index=0}

---

## Features
- **Five recommendation models**: TF-IDF, Doc2Vec, BM25, PyVi-based, and VnCoreNLP-based, implemented for comparison and evaluation. :contentReference[oaicite:1]{index=1}  
- **Vietnamese chatbot** (`/chatbot`) that parses free-form requests, detects ingredients/cooking time/intent, and recommends recipes dynamically. :contentReference[oaicite:2]{index=2}  
- **Model comparison UI** (`/`) to visualize similarities and outputs across all five models. :contentReference[oaicite:3]{index=3}  
- **Optimized training & loading** via cached similarity matrices to speed up experiments. :contentReference[oaicite:4]{index=4}  
- **Corpus growth** using the Spoonacular API to append unique new recipes into `recipes.csv`. :contentReference[oaicite:5]{index=5}

---

## Tech Stack
**Backend & Framework**
- Python 3, Flask

**IR / ML / NLP**
- Gensim (Doc2Vec), Scikit-learn (TF-IDF, cosine similarity), Okapi BM25  
- PyVi (Vietnamese tokenizer), VnCoreNLP (linguistic processing)

**Data & Utilities**
- Pandas, NumPy, Pickle (matrix caching), CSV storage  
- Regex / NLTK / `unicodedata` for text preprocessing  
- HTML/CSS/Bootstrap for front-end templates :contentReference[oaicite:6]{index=6}

---

## Project Structure (high-level)
- `app.py` — Flask app & routes  
- `chatbot.py`, `chatbot_recommender.py` — chatbot logic & integration  
- `recommender_*.py` — per-model recommenders (TF-IDF, Doc2Vec, BM25, PyVi, VnCoreNLP)  
- `model_trainer.py` — training and similarity-matrix generation  
- `recipes.csv` — main recipe corpus  
- `tfidf_improved_matrix.pkl`, `bm25_matrix.pkl`, `vncorenlp_matrix.pkl` — cached matrices  
- `templates/`, `static/` — web UI assets :contentReference[oaicite:7]{index=7}

---

## Getting Started

### 1) Clone
```bash
git clone https://github.com/d3r4nk/foodbot.git
cd foodbot
