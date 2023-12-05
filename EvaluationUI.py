import pandas as pd
from tkinter import ttk
from SearchSystem import search_song_bm25
import tkinter as tk

# Load and Preprocess the Dataset
dataset_path = r"Dataset\test_query.csv"
df_query = pd.read_csv(dataset_path)

# Drop rows with missing lyrics
df_query = df_query.dropna(subset=['corpus'])

# Convert lyrics to lowercase
df_query['corpus'] = df_query['corpus'].str.lower()

# Create a Tkinter GUI for Evaluation Results
class EvaluationResultsApp:
    def __init__(self, master, df_query):
        self.master = master
        self.master.title("Evaluation Results")

        self.label = tk.Label(master, text="Evaluation Results")
        self.label.pack(pady=10)
        self.tree = ttk.Treeview(master)
        self.tree["columns"] = ("Query", "True Labels", "Predicted Labels", "Precision", "Recall", "F1-Score")
        self.tree.column("#0", width=0, stretch=tk.NO)
        self.tree.column("Query", anchor=tk.W, width=200)
        self.tree.column("True Labels", anchor=tk.W, width=200)
        self.tree.column("Predicted Labels", anchor=tk.W, width=200)
        self.tree.column("Precision", anchor=tk.W, width=100)
        self.tree.column("Recall", anchor=tk.W, width=100)
        self.tree.column("F1-Score", anchor=tk.W, width=100)

        self.tree.heading("#0", text="", anchor=tk.W)
        self.tree.heading("Query", text="Query", anchor=tk.W)
        self.tree.heading("True Labels", text="True Labels", anchor=tk.W)
        self.tree.heading("Predicted Labels", text="Predicted Labels", anchor=tk.W)
        self.tree.heading("Precision", text="Precision", anchor=tk.W)
        self.tree.heading("Recall", text="Recall", anchor=tk.W)
        self.tree.heading("F1-Score", text="F1-Score", anchor=tk.W)

        self.tree.pack(pady=10)

        self.evaluate_button = tk.Button(master, text="Evaluate", command=lambda: self.perform_evaluation(df_query))
        self.evaluate_button.pack(pady=10)

        # Bind a function to the treeview selection event
        self.tree.bind("<ButtonRelease-1>", self.show_full_text)
    def perform_evaluation(self, df_query):
        # Clear existing items in the Treeview
        for item in self.tree.get_children():
            self.tree.delete(item)

        precision_list = []
        recall_list = []
        f1_list = []

        for index, row in df_query.iterrows():
            query = row['corpus']
            true_labels = [(row['title'], row['artist'])]

            search_results = search_song_bm25(query)
            predicted_labels = list(zip(search_results['title'], search_results['artist']))

            true_positives = sum(1 for true_label in true_labels if true_label in predicted_labels)
            false_positives = sum(1 for pred_label in predicted_labels if pred_label not in true_labels)
            false_negatives = sum(1 for true_label in true_labels if true_label not in predicted_labels)

            precision_query = true_positives / (
                        true_positives + false_positives) if true_positives + false_positives > 0 else 0
            recall_query = true_positives / (
                        true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
            f1_query = 2 * (precision_query * recall_query) / (
                        precision_query + recall_query) if precision_query + recall_query > 0 else 0

            self.tree.insert("", "end",
                             values=(query, true_labels, predicted_labels, precision_query, recall_query, f1_query))

            precision_list.append(precision_query)
            recall_list.append(recall_query)
            f1_list.append(f1_query)

        # Average metrics
        precision_avg = sum(precision_list) / len(precision_list)
        recall_avg = sum(recall_list) / len(recall_list)
        f1_avg = sum(f1_list) / len(f1_list)

        # Display average metrics as the last row in the table
        self.tree.insert("", "end", values=("Average Metrics", "", "", precision_avg, recall_avg, f1_avg))

    def show_full_text(self, event):
        # Get the selected item in the treeview
        selected_item = self.tree.selection()
        if selected_item:
            # Extract the information from the selected item
            values = self.tree.item(selected_item, 'values')
            # Create a new window to display the full text of each column
            self.show_details_window(values)

    def show_details_window(self, values):
        details_window = tk.Toplevel(self.master)
        details_window.title("Query Details")

        labels = ["Query", "True Labels", "Predicted Labels", "Precision", "Recall", "F1-Score"]
        for label, value in zip(labels, values):
            label_widget = tk.Label(details_window, text=f"{label}: {value}")
            label_widget.pack(pady=5)
# Run the Tkinter GUI for Evaluation Results
root = tk.Tk()
app = EvaluationResultsApp(root, df_query)
root.mainloop()