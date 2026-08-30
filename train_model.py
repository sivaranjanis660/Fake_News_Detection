import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================================================
# 1. LOAD DATASET
# =========================================================

fake = pd.read_csv("DATASET/Fake.csv")
true = pd.read_csv("DATASET/True.csv")

print("Dataset loaded successfully!")


# =========================================================
# 2. ADD LABELS
# =========================================================

fake["label"] = 0       # Fake
true["label"] = 1       # Real


# =========================================================
# 3. COMBINE DATASETS
# =========================================================

data = pd.concat([fake, true], ignore_index=True)

print("\nOriginal dataset shape:", data.shape)


# =========================================================
# 4. CREATE CONTENT
# =========================================================

data["title"] = data["title"].fillna("").astype(str)
data["text"] = data["text"].fillna("").astype(str)

data["content"] = data["title"] + " " + data["text"]


# =========================================================
# 5. REMOVE EMPTY ARTICLES
# =========================================================

data = data[data["content"].str.strip() != ""]


# =========================================================
# 6. REMOVE DUPLICATES
# =========================================================

data = data.drop_duplicates(subset=["content"])


# =========================================================
# 7. SHUFFLE DATA
# =========================================================

data = data.sample(frac=1, random_state=42).reset_index(drop=True)


# =========================================================
# 8. DISPLAY DATASET INFORMATION
# =========================================================

print("\nDataset after cleaning:", data.shape)

print("\nFake news:", (data["label"] == 0).sum())
print("Real news:", (data["label"] == 1).sum())


# =========================================================
# 9. INPUT AND OUTPUT
# =========================================================

X = data["content"]
y = data["label"]


# =========================================================
# 10. TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================================================
# 11. TF-IDF VECTORIZATION
# =========================================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=100000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.98,
    sublinear_tf=True
)

print("\nConverting text into TF-IDF features...")

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF training shape:", X_train_tfidf.shape)
print("TF-IDF testing shape:", X_test_tfidf.shape)


# =========================================================
# 12. TRAIN LOGISTIC REGRESSION MODEL
# =========================================================

model = LogisticRegression(
    C=2.0,
    max_iter=2000,
    class_weight="balanced",
    solver="liblinear"
)

print("\nTraining model...")

model.fit(X_train_tfidf, y_train)

print("Model training completed!")


# =========================================================
# 13. PREDICTION
# =========================================================

y_pred = model.predict(X_test_tfidf)


# =========================================================
# 14. MODEL ACCURACY
# =========================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n====================================")
print("MODEL PERFORMANCE")
print("====================================")

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")


# =========================================================
# 15. CLASSIFICATION REPORT
# =========================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Fake News", "Real News"]
    )
)


# =========================================================
# 16. CONFUSION MATRIX
# =========================================================

print("\nConfusion Matrix:")

cm = confusion_matrix(y_test, y_pred)

print(cm)


# =========================================================
# 17. SAVE MODEL
# =========================================================

joblib.dump(model, "fake_news_model.pkl")

joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\n====================================")
print("Model saved successfully!")
print("fake_news_model.pkl")
print("tfidf_vectorizer.pkl")
print("====================================")
