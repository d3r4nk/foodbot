import pandas as pd
import os
import pickle
import time
from recommender_doc2vec import build_recommender as build_doc2vec
from recommender_tf_idf import build_recommender as build_tfidf
from recommender_pyvi import build_recommender as build_pyvi
from recommender_bm25 import build_recommender as build_bm25
from recommender_vncorenlp import build_recommender as build_vncorenlp

def preprocess_data(df):
    #tiền xử lý dữ liệu 
    df['soup'] = df['title'] + ' ' + df['ingredients'] + ' ' + df['instructions']
    df['soup'] = df['soup'].fillna('').astype(str)
    return df
def build_all_models(force_rebuild=False):
    # Tạo thư mục cache nếu chưa tồn tại , sài cache để lưu mô hình và ma trận tương đồng 
    if not os.path.exists('model_cache'):
        os.makedirs('model_cache')
    # Đọc dữ liệu
    df = preprocess_data(pd.read_csv('recipes.csv'))
    # Danh sách các mô hình cần xây dựng
    models = [
        ('doc2vec', build_doc2vec, 'model_cache/doc2vec_matrix.pkl'),
        ('tfidf', build_tfidf, 'model_cache/tfidf_matrix.pkl'),
        ('pyvi', build_pyvi, 'model_cache/pyvi_matrix.pkl'),
        ('bm25', build_bm25, 'model_cache/bm25_matrix.pkl'),
        ('vncorenlp', build_vncorenlp, 'model_cache/vncorenlp_matrix.pkl'),
    ]
    #build và lưu các mô hình 
    for model_name, build_function, cache_path in models:
        if not os.path.exists(cache_path) or force_rebuild:
            start_time = time.time()
            # Xây dựng mô hình
            sim_matrix = build_function(df)
            # Lưu ma trận tương đồng
            with open(cache_path, 'wb') as f:
                pickle.dump(sim_matrix, f)
            end_time = time.time()
        else:
            print(f"Mô hình {model_name} đã tồn tại tại {cache_path}.")  #nếu có sẵn model rồi thì xuất ra thông báo nàynày
    print("Đã hoàn thành việc xây dựng tất cả các mô hình.")

def load_all_models():
    #tải hết model lên và đọc dữ liệu 
    df = preprocess_data(pd.read_csv('recipes.csv'))
    print(f"Đã tải {len(df)} công thức món ăn.")  #load các món ăn lên app 
    # Danh sách các mô hình cần tải
    model_paths = [
        ('doc2vec', 'model_cache/doc2vec_matrix.pkl'),
        ('tfidf', 'model_cache/tfidf_matrix.pkl'),
        ('pyvi', 'model_cache/pyvi_matrix.pkl'),
        ('bm25', 'model_cache/bm25_matrix.pkl'),
        ('vncorenlp', 'model_cache/vncorenlp_matrix.pkl'),
    ]
    #tải các mô hình 
    similarity_matrices = {}
    for model_name, cache_path in model_paths:
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                similarity_matrices[model_name] = pickle.load(f)
        else:
            print(f"Cảnh báo: Không tìm thấy mô hình {model_name} tại {cache_path}.")
            #không thấy model thì xuất ra thông báo này 
    return df, similarity_matrices
if __name__ == "__main__":
    #xây dựng tất cả mô hình chỉ 1 lần duy nhất
    build_all_models(force_rebuild=False)