import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. LOAD DATASETS
# ==========================================

fake = pd.read_csv("DATASET/Fake.csv")
true = pd.read_csv("DATASET/True.csv")

print("Datasets loaded successfully!")


# ==========================================
# 2. ADD LABELS
# ==========================================

# 0 = Fake News
# 1 = Real News

fake["label"] = 0
true["label"] = 1


# ==========================================
# 3. COMBINE DATASETS
# ==========================================

data = pd.concat([fake, true], ignore_index=True)


print("\nDataset shape before cleaning:", data.shape)


# ==========================================
# 4. CREATE CONTENT
# ==========================================

data["content"] = (
    data["title"].fillna("").astype(str)
    + " "
    + data["text"].fillna("").astype(str)
)


# ==========================================
# 5. REMOVE EMPTY CONTENT
# ==========================================

data = data[data["content"].str.strip() != ""]


# ==========================================
# 6. REMOVE DUPLICATE NEWS
# ==========================================

data = data.drop_duplicates(subset=["content"])


# ==========================================
# 7. SHUFFLE DATASET
# ==========================================

data = data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ==========================================
# 8. DISPLAY DATASET INFORMATION
# ==========================================

print("\nDataset shape after cleaning:", data.shape)

print("\nTotal News:", len(data))

print("Fake News:", sum(data["label"] == 0))

print("Real News:", sum(data["label"] == 1))


# ==========================================
# 9. SELECT INPUT AND OUTPUT
# ==========================================

X = data["content"]
y = data["label"]


# ==========================================
# 10. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining data:", len(X_train))
print("Testing data:", len(X_test))


# ==========================================
# 11. TF-IDF VECTORIZER
# ==========================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=100000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.98,
    sublinear_tf=True
)


# Convert text into numerical values

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)


print("\nTF-IDF conversion completed!")

print("Number of features:", X_train_tfidf.shape[1])


# ==========================================
# 12. CREATE LOGISTIC REGRESSION MODEL
# ==========================================

model = LogisticRegression(
    C=2.0,
    max_iter=1000,
    class_weight="balanced"
)


# ==========================================
# 13. TRAIN MODEL
# ==========================================

print("\nTraining the model...")

model.fit(X_train_tfidf, y_train)

print("Model training completed!")


# ==========================================
# 14. PREDICT TEST DATA
# ==========================================

y_pred = model.predict(X_test_tfidf)


# ==========================================
# 15. CALCULATE ACCURACY
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n===================================")
print("MODEL PERFORMANCE")
print("===================================")

print("Model Accuracy:", round(accuracy * 100, 2), "%")


# ==========================================
# 16. CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Fake News", "Real News"]
    )
)


# ==========================================
# 17. SAVE TRAINED MODEL
# ==========================================

joblib.dump(
    model,
    "fake_news_model.pkl"
)

joblib.dump(
    vectorizer,
    "tfidf_vectorizer.pkl"
)


# ==========================================
# 18. FINAL MESSAGE
# ==========================================

print("\n===================================")
print("MODEL SAVED SUCCESSFULLY!")
print("===================================")

print("fake_news_model.pkl created")
print("tfidf_vectorizer.pkl created")
