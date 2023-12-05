import pandas as pd
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from SearchSystem import search_song_bm25

# Step 1: Load and Preprocess the Dataset
dataset_path = r"Dataset\test_query.csv"  # Replace with the actual path
df_query = pd.read_csv(dataset_path)

# Drop rows with missing lyrics
df_query = df_query.dropna(subset=['corpus'])

# Convert lyrics to lowercase
df_query['corpus'] = df_query['corpus'].str.lower()

# Remove stopwords and tokenize lyrics
stop_words = set(stopwords.words('english'))
df_query['corpus'] = df_query['corpus'].apply(
    lambda x: ' '.join([word for word in word_tokenize(x) if word.isalnum() and word not in stop_words]))

# Step 2: BM25 Vectorization
corpus = df_query['corpus'].apply(str.split).tolist()
bm25_query = BM25Okapi(corpus)

# Step 3: Evaluation Function
def evaluate_search_system(df_query, bm25_query):
    precision_list = []
    recall_list = []
    f1_list = []

    for index, row in df_query.iterrows():
        query = row['corpus']
        true_labels = [(row['title'], row['artist'])]

        search_results = search_song_bm25(query)
        predicted_labels = list(zip(search_results['title'], search_results['artist']))

        # Calculate precision, recall, and F1-score for the current query
        true_positives = sum(1 for true_label in true_labels if true_label in predicted_labels)
        false_positives = sum(1 for pred_label in predicted_labels if pred_label not in true_labels)
        false_negatives = sum(1 for true_label in true_labels if true_label not in predicted_labels)

        precision_query = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        recall_query = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
        f1_query = 2 * (precision_query * recall_query) / (precision_query + recall_query) if precision_query + recall_query > 0 else 0

        # Print the result for each query
        print(f"\nQuery: {query}")
        print(f"Precision: {precision_query:.4f}")
        print(f"Recall: {recall_query:.4f}")
        print(f"F1-Score: {f1_query:.4f}")

        precision_list.append(precision_query)
        recall_list.append(recall_query)
        f1_list.append(f1_query)

    # Average precision, recall, and F1-score over all queries
    precision_avg = sum(precision_list) / len(precision_list)
    recall_avg = sum(recall_list) / len(recall_list)
    f1_avg = sum(f1_list) / len(f1_list)

    print("\nAverage Metrics:")
    print(f"Precision: {precision_avg:.4f}")
    print(f"Recall: {recall_avg:.4f}")
    print(f"F1-Score: {f1_avg:.4f}")

# Example Usage
evaluate_search_system(df_query, bm25_query)
