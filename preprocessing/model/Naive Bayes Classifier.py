import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

def main():
    data_path = "data/komentar_bersih.csv"
    model_dir = "model"

    if not os.path.exists(data_path):
        print(f"[ERROR] File tidak ditemukan: {data_path}")
        return

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"[ERROR] Gagal membaca file CSV: {e}")
        return

    if "komentar_bersih" not in df.columns or "label" not in df.columns:
        print("[ERROR] Kolom 'komentar_bersih' atau 'label' tidak ditemukan.")
        return

    df = df.dropna(subset=["komentar_bersih", "label"])
    df = df[df['label'].isin(['positif', 'negatif', 'netral'])]

    min_count = df['label'].value_counts().min()
    df = df.groupby('label').apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)

    print("\n[INFO] Distribusi label setelah balancing:")
    print(df['label'].value_counts())

    X_text = df["komentar_bersih"].astype(str)
    y = df["label"].astype(str)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=2,
        sublinear_tf=True,
        stop_words='indonesian'
    )
    X = vectorizer.fit_transform(X_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Naive Bayes
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Simpan model dan vectorizer
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(model_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(model, os.path.join(model_dir, "naive_bayes_model.pkl"))

    # Evaluasi
    y_pred = model.predict(X_test)

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
