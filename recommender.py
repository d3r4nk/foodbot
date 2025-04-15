# recommender.py
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data(path='recipes.csv'):
    df = pd.read_csv(path)
    df.fillna('', inplace=True)
    df['soup'] = df['title'] + ' ' + df['ingredients']
    return df

def build_recommender(df):
    count = CountVectorizer(stop_words='english')
    matrix = count.fit_transform(df['soup'])
    sim = cosine_similarity(matrix, matrix)
    return sim

def recommend(df, sim, title):
    indices = pd.Series(df.index, index=df['title'])
    idx = indices.get(title)
    if idx is None:
        return []
    scores = list(enumerate(sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
    result_indices = [i[0] for i in scores]
    return df.iloc[result_indices]
