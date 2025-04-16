import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_recommender(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['soup'].fillna(''))
    sim_matrix = cosine_similarity(tfidf_matrix)
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
