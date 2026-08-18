from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model and TF-IDF vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"]

    # Convert news text into TF-IDF features
    news_tfidf = vectorizer.transform([news])

    # Make prediction
    prediction = model.predict(news_tfidf)[0]

    # Get prediction probability
    probability = model.predict_proba(news_tfidf)[0]

    if prediction == 0:
        result = "FAKE NEWS"
        confidence = round(probability[0] * 100, 2)
    else:
        result = "REAL NEWS"
        confidence = round(probability[1] * 100, 2)

    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)