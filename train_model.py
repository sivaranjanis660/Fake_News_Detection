import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =====================================================
# 1. LOAD DATASET
# =====================================================

print("Loading datasets...")

fake = pd.read_csv("DATASET/Fake.csv")
true = pd.read_csv("DATASET/True.csv")

print("Fake dataset:", fake.shape)
print("Real dataset:", true.shape)


# =====================================================
# 2. ADD LABELS
# =====================================================

fake["label"] = 0
true["label"] = 1


# =====================================================
# 3. COMBINE DATASETS
# =====================================================

data = pd.concat([fake, true], ignore_index=True)

print("\nCombined dataset:", data.shape)


# =====================================================
# 4. HANDLE MISSING VALUES
# =====================================================

data["title"] = data["title"].fillna("").astype(str)
data["text"] = data["text"].fillna("").astype(str)


# =====================================================
# 5. CREATE CONTENT
# =====================================================

data["content"] = data["title"] + " " + data["text"]


# =====================================================
# 6. REMOVE EMPTY CONTENT
# =====================================================

data = data[data["content"].str.strip() != ""]


# =====================================================
# 7. REMOVE DUPLICATES
# =====================================================

data = data.drop_duplicates(subset=["content"])


# =====================================================
# 8. SHUFFLE DATA
# =====================================================

data = data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# =====================================================
# 9. DISPLAY DATA INFORMATION
# =====================================================

print("\nAfter cleaning:", data.shape)

print("Fake news:", (data["label"] == 0).sum())
print("Real news:", (data["label"] == 1).sum())


# =====================================================
# 10. INPUT AND OUTPUT
# =====================================================

X = data["content"]
y = data["label"]


# =====================================================
# 11. TRAIN / TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# =====================================================
# 12. TF-IDF
# =====================================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=100000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.98,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF training shape:", X_train_tfidf.shape)
print("TF-IDF testing shape:", X_test_tfidf.shape)


# =====================================================
# 13. TRAIN MODEL
# =====================================================

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    C=2.0,
    max_iter=2000,
    class_weight="balanced",
    solver="liblinear"
)

model.fit(X_train_tfidf, y_train)

print("Training completed!")


# =====================================================
# 14. PREDICTION
# =====================================================

y_pred = model.predict(X_test_tfidf)


# =====================================================
# 15. ACCURACY
# =====================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n======================================")
print("MODEL PERFORMANCE")
print("======================================")

print(
    "Model Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


# =====================================================
# 16. CLASSIFICATION REPORT
# =====================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Fake News",
            "Real News"
        ]
    )
)


# =====================================================
# 17. CONFUSION MATRIX
# =====================================================

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# =====================================================
# 18. SAVE MODEL
# =====================================================

print("\nSaving model...")

joblib.dump(
    model,
    "fake_news_model.pkl"
)

joblib.dump(
    vectorizer,
    "tfidf_vectorizer.pkl"
)


print("\n======================================")
print("MODEL SAVED SUCCESSFULLY!")
print("======================================")
print("fake_news_model.pkl")
print("tfidf_vectorizer.pkl")
