from flask import Flask, render_template, request
import pandas as pd
from recommender import clean_html
from recommender_doc2vec import build_recommender as build_doc2vec, recommend as recommend_doc2vec
from recommender_spacy import build_recommender as build_spacy, recommend as recommend_spacy
from recommender_tf_idf import build_recommender as build_tfidf, recommend as recommend_tfidf

app = Flask(__name__)

# Tiền xử lý dữ liệu
def preprocess_data(df):
    df['soup'] = df['title'] + ' ' + df['ingredients'] + ' ' + df['instructions']
    df['soup'] = df['soup'].fillna('').astype(str)
    return df

# Load dữ liệu 1 lần
df = preprocess_data(pd.read_csv('recipes.csv'))

# Cache ma trận tương đồng cho mỗi mô hình
similarity_matrices = {
    'doc2vec': build_doc2vec(df),
    'spacy': build_spacy(df),
    'tfidf': build_tfidf(df)
}

@app.route('/', methods=['GET', 'POST'])
def index():
    recipes = df[['title']].to_dict(orient='records')
    selected_title = None
    recommendations = []
    selected_method = 'doc2vec'

    if request.method == 'POST' and 'title' in request.form:
        selected_title = request.form['title']
        selected_method = request.form.get('method', 'doc2vec')

        # Gọi hàm tương ứng với mô hình
        sim_matrix = similarity_matrices[selected_method]
        if selected_method == 'doc2vec':
            recommendations = recommend_doc2vec(df, sim_matrix, selected_title)
        elif selected_method == 'spacy':
            recommendations = recommend_spacy(df, sim_matrix, selected_title)
        elif selected_method == 'tfidf':
            recommendations = recommend_tfidf(df, sim_matrix, selected_title)

        # Clean HTML
        for rec in recommendations:
            rec['instructions'] = clean_html(rec.get('instructions', ''))
            rec['ingredients'] = clean_html(rec.get('ingredients', ''))
    
    return render_template('index.html', recipes=recipes, selected=selected_title, recommendations=recommendations, selected_method=selected_method)

if __name__ == '__main__':
    app.run(debug=True)
