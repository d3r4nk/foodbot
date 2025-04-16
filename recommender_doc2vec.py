import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Tải tokenizer nếu chưa có
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def build_recommender(df):
    texts = df['soup'].fillna('').astype(str).tolist()
    tagged_data = [TaggedDocument(words=word_tokenize(doc.lower()), tags=[str(i)]) for i, doc in enumerate(texts)]
    model = Doc2Vec(vector_size=100, window=5, min_count=2, workers=4, epochs=40)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    vectors = [model.infer_vector(word_tokenize(text.lower())) for text in texts]
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
