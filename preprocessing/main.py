import pandas as pd
from preprocessing.text_cleaner import preprocess_text
from textblob import TextBlob

def detect_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "positif"
    elif polarity < -0.1:
        return "negatif"
    else:
        return "netral"

df = pd.read_csv("data/komentar_instagram.csv")

df['komentar_bersih'] = df['komentar'].apply(preprocess_text)

df['label'] = df['komentar_bersih'].apply(detect_sentiment)

# Simpan hasil ke file
df.to_csv("data/komentar_bersih.csv", index=False)
print(df[['komentar', 'komentar_bersih', 'label']].head())
