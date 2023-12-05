import pandas as pd
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import tkinter as tk
from tkinter import Scrollbar, Listbox, Entry, Button, Label, Tk

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

# Create a Tkinter GUI
class MusicSearchApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Music Search App")

        self.label = Label(master, text="Enter the lyrics or a phrase:")
        self.label.pack(pady=10)

        self.entry = Entry(master, width=50)
        self.entry.pack(pady=10)

        self.search_button = Button(master, text="Search", command=self.perform_search)
        self.search_button.pack(pady=10)

        self.results_listbox = Listbox(master, width=60, height=10)
        self.results_listbox.pack(pady=10)

        self.scrollbar = Scrollbar(master, orient="vertical")
        self.scrollbar.config(command=self.results_listbox.yview)
        self.scrollbar.pack(side="right", fill="y")

        self.results_listbox.config(yscrollcommand=self.scrollbar.set)

    def perform_search(self):
        user_query = self.entry.get()
        search_results = search_song_bm25(user_query)

        self.results_listbox.delete(0, tk.END)  # Clear previous results

        for index, row in search_results.iterrows():
            result_str = f"{row['title']} - {row['artist']}"
            self.results_listbox.insert(tk.END, result_str)

# Run the Tkinter GUI
root = Tk()
app = MusicSearchApp(root)
root.mainloop()
