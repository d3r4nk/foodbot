from flask import Flask, render_template, request
from recommender import load_data, build_recommender, recommend

app = Flask(__name__)

df = load_data()
sim = build_recommender(df)

@app.route("/", methods=["GET", "POST"])
def index():
    suggestions = None
    selected = None

    if request.method == "POST":
        selected = request.form.get("title")
        if selected:
            suggestions = recommend(df, sim, selected)

    return render_template("index.html", recipes=df['title'].tolist(), suggestions=suggestions, selected=selected)

if __name__ == "__main__":
    app.run(debug=True)
