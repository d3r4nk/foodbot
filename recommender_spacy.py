import spacy
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Tải mô hình tiếng Anh của spaCy nếu chưa có
try:
    nlp = spacy.load("en_core_web_md")
except:
    import os
    os.system("python -m spacy download en_core_web_md")
    nlp = spacy.load("en_core_web_md")

def build_recommender(df):
    docs = df['soup'].fillna('').astype(str).tolist()
    vectors = [nlp(doc).vector for doc in docs]
    sim_matrix = cosine_similarity(vectors)
    return sim_matrix

def recommend(df, sim_matrix, title):
    indices = pd.Series(df.index, index=df['title'])
    idx = indices.get(title)
    if idx is None:
        return []
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
    result_indices = [i[0] for i in scores]
    return df.iloc[result_indices].to_dict(orient='records')
