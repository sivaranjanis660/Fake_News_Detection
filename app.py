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

    # Get news from user
    news = request.form.get("news", "").strip()

    # Check empty input
    if not news:
        return render_template(
            "index.html",
            prediction="Please enter some news.",
            confidence=None
        )

    # Check very short input
    word_count = len(news.split())

    if word_count < 10:
        return render_template(
            "index.html",
            prediction="Please enter a complete news article.",
            confidence=None
        )

    # Convert news into TF-IDF features
    news_tfidf = vectorizer.transform([news])

    # Make prediction
    prediction = model.predict(news_tfidf)[0]

    # Get prediction probability
    probability = model.predict_proba(news_tfidf)[0]

    max_probability = max(probability)
    confidence = round(max_probability * 100, 2)

    if max_probability < 0.65:
       result = "UNCERTAIN - PLEASE VERIFY"
    elif prediction == 0:
       result = "FAKE NEWS"
    else:
       result = "REAL NEWS"

    # Display result
    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)
