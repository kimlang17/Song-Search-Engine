import pandas as pd
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load and Preprocess the Dataset
dataset_path = r"Dataset\music_data.csv"
df = pd.read_csv(dataset_path)

# Drop rows with missing lyrics
df = df.dropna(subset=['lyrics'])

# Convert lyrics to lowercase
df['lyrics'] = df['lyrics'].str.lower()

# Remove stopwords and tokenize lyrics
stop_words = set(stopwords.words('english'))
df['lyrics'] = df['lyrics'].apply(
    lambda x: ' '.join([word for word in word_tokenize(x) if word.isalnum() and word not in stop_words]))

# BM25 Vectorization
corpus = df['lyrics'].apply(str.split).tolist()
bm25 = BM25Okapi(corpus)

# Search Function
def search_song_bm25(query):
    query = query.lower()
    query = [word for word in word_tokenize(query) if word.isalnum() and word not in stop_words]

    scores = bm25.get_scores(query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]

    results = df.iloc[top_indices][['title', 'artist']]
    return results