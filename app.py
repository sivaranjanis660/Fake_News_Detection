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

    # Get news text
    news = request.form.get("news", "").strip()

    # Empty input check
    if not news:
        return render_template(
            "index.html",
            prediction="Please enter some news.",
            confidence=None
        )

    # Short input check
    word_count = len(news.split())

    if word_count < 10:
        return render_template(
            "index.html",
            prediction="Please enter a complete news article.",
            confidence=None
        )

    # Convert text to TF-IDF
    news_tfidf = vectorizer.transform([news])

    # Prediction
    prediction = model.predict(news_tfidf)[0]

    # Probability
    probability = model.predict_proba(news_tfidf)[0]

    # Highest probability
    max_probability = max(probability)

    # Confidence percentage
    confidence = round(max_probability * 100, 2)

    # Result logic
    if confidence < 45:
        result = "FAKE NEWS"

    elif confidence <= 65:
        result = "UNCERTAIN - PLEASE VERIFY"

    else:
        if prediction == 0:
            result = "FAKE NEWS"
        else:
            result = "REAL NEWS"

    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)
