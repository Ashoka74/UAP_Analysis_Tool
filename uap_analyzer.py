import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap
# import hdbscan
import fast_hdbscan as hdbscan
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
import torch
with torch.no_grad():
    embed_model = SentenceTransformer('all-mpnet-base-v2')
    embed_model.to('cuda')
from sentence_transformers.util import pytorch_cos_sim, pairwise_cos_sim
#from stqdm.notebook import stqdm
#stqdm.pandas()
import logging
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap
import hdbscan
import plotly.graph_objects as go
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from Levenshtein import distance
import logging
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic
from statsmodels.api import stats
import os
import time
import concurrent.futures
from requests.exceptions import HTTPError
from stqdm import stqdm
stqdm.pandas()
import json
import pandas as pd
from openai import OpenAI
import numpy as np
import matplotlib.pyplot as plt
import squarify
import matplotlib.colors as mcolors
import textwrap
import pandas as pd
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class UAPAnalyzer:
    """
    A class for analyzing and clustering textual data within a pandas DataFrame using 
    Natural Language Processing (NLP) techniques and machine learning models.
    
    Attributes:
        data (pd.DataFrame): The dataset containing textual data for analysis.
        column (str): The name of the column in the DataFrame to be analyzed.
        embeddings (np.ndarray): The vector representations of textual data.
        reduced_embeddings (np.ndarray): The dimensionality-reduced embeddings.
        cluster_labels (np.ndarray): The labels assigned to each data point after clustering.
        cluster_terms (list): The list of terms associated with each cluster.
        tfidf_matrix (sparse matrix): The Term Frequency-Inverse Document Frequency (TF-IDF) matrix.
        models (dict): A dictionary to store trained machine learning models.
        evaluations (dict): A dictionary to store evaluation results of models.
        data_nums (pd.DataFrame): The DataFrame with numerical encoding of categorical data.
    """

    def __init__(self, data, column, has_embeddings=False):
        """
        Initializes the UAPAnalyzer with a dataset and a specified column for analysis.
        
        Args:
            data (pd.DataFrame): The dataset for analysis.
            column (str): The column within the dataset to analyze.
        """
        assert isinstance(data, pd.DataFrame), "Data must be a pandas DataFrame"
        assert column in data.columns, f"Column '{column}' not found in DataFrame"
        self.has_embeddings = has_embeddings        
        self.data = data
        self.column = column
        self.embeddings = None
        self.reduced_embeddings = None
        self.cluster_labels = None
        self.cluster_names = None
        self.cluster_terms = None 
        self.cluster_terms_embeddings = None
        self.tfidf_matrix = None
        self.models = {}  # To store trained models
        self.evaluations = {}  # To store evaluation results
        self.data_nums = None  # Encoded numerical data
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.preds = None
        self.new_dataset = None
        self.model = SentenceTransformer('embaas/sentence-transformers-e5-large-v2')
        self.model = self.model.to('cuda')
        #self.cluster_names_ = pd.DataFrame()

        logging.info("UAPAnalyzer initialized")

    def preprocess_data(self, trim=False, has_embeddings=False, top_n=32,):
        """
        Preprocesses the data by optionally trimming the dataset to include only the top N labels and extracting embeddings.
        
        Args:
            trim (bool): Whether to trim the dataset to include only the top N labels.
            top_n (int): The number of top labels to retain if trimming is enabled.
        """
        logging.info("Preprocessing data")

        # if trim is True
        if trim:
            # Identify the top labels based on value counts
            top_labels = self.data[self.column].value_counts().nlargest(top_n).index.tolist()
            # Revise the column data, setting values to 'Other' if they are not in the top labels
            self.data[f'{self.column}_revised'] = np.where(self.data[self.column].isin(top_labels), self.data[self.column], 'Other')
        # Convert the column data to string type before passing to _extract_embeddings
        # This is useful especially if the data type of the column is not originally string
        string_data = self.data[f'{self.column}'].astype(str)
        # Extract embeddings from the revised and string-converted column data
        if has_embeddings:
            self.embeddings = self.data['embeddings'].to_list()
        else:
            self.embeddings = self._extract_embeddings(string_data)
        logging.info("Data preprocessing complete")


    def _extract_embeddings(self, data_column):
        """
        Extracts embeddings from the given data column.
        
        Args:
            data_column (pd.Series): The column from which to extract embeddings.
        
        Returns:
            np.ndarray: The extracted embeddings.
        """
        logging.info("Extracting embeddings")
        # convert to str
        return embed_model.encode(data_column.tolist(), show_progress_bar=True)

    def reduce_dimensionality(self, method='UMAP', n_components=2, **kwargs):
        """
        Reduces the dimensionality of embeddings using specified method.
        
        Args:
            method (str): The dimensionality reduction method to use ('UMAP' or 'PCA').
            n_components (int): The number of dimensions to reduce to.
            **kwargs: Additional keyword arguments for the dimensionality reduction method.
        """
        logging.info(f"Reducing dimensionality using {method}")
        if method == 'UMAP':
            reducer = umap.UMAP(n_components=n_components, **kwargs)
        elif method == 'PCA':
            reducer = PCA(n_components=n_components)
        else:
            raise ValueError("Unsupported dimensionality reduction method")
        
        self.reduced_embeddings = reducer.fit_transform(self.embeddings)
        logging.info(f"Dimensionality reduced using {method}")

    def cluster_data(self, method='HDBSCAN', **kwargs):
        """
        Clusters the reduced dimensionality data using the specified clustering method.
        
        Args:
            method (str): The clustering method to use ('HDBSCAN' or 'KMeans').
            **kwargs: Additional keyword arguments for the clustering method.
        """
        logging.info(f"Clustering data using {method}")
        if method == 'HDBSCAN':
            clusterer = hdbscan.HDBSCAN(**kwargs)
        elif method == 'KMeans':
            clusterer = KMeans(**kwargs)
        else:
            raise ValueError("Unsupported clustering method")
        
        clusterer.fit(self.reduced_embeddings)
        self.cluster_labels = clusterer.labels_
        logging.info(f"Data clustering complete using {method}")

        
    def get_tf_idf_clusters(self, top_n=2):
        """
        Names clusters using the most frequent terms based on TF-IDF analysis.

        Args:
            top_n (int): The number of top terms to consider for naming each cluster.
        """
        logging.info("Naming clusters based on top TF-IDF terms.")

        # Ensure data has been clustered
        assert self.cluster_labels is not None, "Data has not been clustered yet."
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

        # Fit the vectorizer to the text data and transform it into a TF-IDF matrix
        tfidf_matrix = vectorizer.fit_transform(self.data[f'{self.column}'].astype(str))

        # Initialize an empty list to store the cluster terms
        self.cluster_terms = []

        for cluster_id in np.unique(self.cluster_labels):
            # Skip noise if present (-1 in HDBSCAN)
            if cluster_id == -1:
                continue

            # Find indices of documents in the current cluster
            indices = np.where(self.cluster_labels == cluster_id)[0]

            # Compute the mean TF-IDF score for each term in the cluster
            cluster_tfidf_mean = np.mean(tfidf_matrix[indices], axis=0)

            # Use the matrix directly for indexing if it does not support .toarray()
            # Ensure it's in a format that supports indexing, convert if necessary
            if hasattr(cluster_tfidf_mean, "toarray"):
                dense_mean = cluster_tfidf_mean.toarray().flatten()
            else:
                dense_mean = np.asarray(cluster_tfidf_mean).flatten()

            # Get the indices of the top_n terms
            top_n_indices = np.argsort(dense_mean)[-top_n:]

            # Get the corresponding terms for these top indices
            terms = vectorizer.get_feature_names_out()
            top_terms = [terms[i] for i in top_n_indices]

            # Join the top_n terms with a hyphen
            cluster_name = '-'.join(top_terms)

            # Append the cluster name to the list
            self.cluster_terms.append(cluster_name)

        # Convert the list of cluster terms to a categorical data type
        self.cluster_terms = pd.Categorical(self.cluster_terms)
        logging.info("Cluster naming completed.")

    def merge_similar_clusters(self, distance='cosine', char_diff_threshold = 3, similarity_threshold = 0.92, embeddings = 'SBERT'):
        """
        Merges similar clusters based on cosine similarity of their associated terms.
        
        Args:
            similarity_threshold (float): The similarity threshold above which clusters are considered similar enough to merge.
        """
        from collections import defaultdict
        logging.info("Merging similar clusters")

        # A mapping from cluster names to a set of cluster names to be merged
        merge_mapping = defaultdict(set)
        merge_labels = defaultdict(set)

        if distance == 'levenshtein':
            distances = {}
            for i, name1 in enumerate(self.cluster_terms):
                for j, name2 in enumerate(self.cluster_terms[i + 1:], start=i + 1):
                    dist = distance(name1, name2)
                    if dist <= char_diff_threshold:
                        logging.info(f"Merging '{name2}' into '{name1}'")
                        merge_mapping[name1].add(name2)

        elif distance == 'cosine':
            self.cluster_terms_embeddings = embed_model.encode(self.cluster_terms)
            cos_sim_matrix = pytorch_cos_sim(self.cluster_terms_embeddings, self.cluster_terms_embeddings)
            for i, name1 in enumerate(self.cluster_terms):
                for j, name2 in enumerate(self.cluster_terms[i + 1:], start=i + 1):
                    if cos_sim_matrix[i][j] > similarity_threshold:
                        #st.write(f"Merging cluster '{name2}' into cluster '{name1}' based on cosine similarity")
                        logging.info(f"Merging cluster '{name2}' into cluster '{name1}' based on cosine similarity")
                        merge_mapping[name1].add(name2)


        # Flatten the merge mapping to a simple name change mapping
        name_change_mapping = {}
        for cluster_name, merges in merge_mapping.items():
            for merge_name in merges:
                name_change_mapping[merge_name] = cluster_name

        # Update cluster labels based on name changes
        updated_cluster_terms = []
        original_to_updated_index = {}
        for i, name in enumerate(self.cluster_terms):
            updated_name = name_change_mapping.get(name, name)
            if updated_name not in updated_cluster_terms:
                updated_cluster_terms.append(updated_name)
                original_to_updated_index[i] = len(updated_cluster_terms) - 1
            else:
                updated_index = updated_cluster_terms.index(updated_name)
                original_to_updated_index[i] = updated_index

        self.cluster_terms = updated_cluster_terms  # Update cluster terms with merged names
        self.clusters_labels = np.array([original_to_updated_index[label] for label in self.cluster_labels])


        # Update cluster labels according to the new index mapping
        # self.cluster_labels = np.array([original_to_updated_index[label] if label in original_to_updated_index else -1 for label in self.cluster_labels])
        # self.cluster_terms = [self.cluster_terms[original_to_updated_index[label]] if label != -1 else 'Noise' for label in self.cluster_labels]

        # Log the total number of merges
        total_merges = sum(len(merges) for merges in merge_mapping.values())
        logging.info(f"Total clusters merged: {total_merges}")

        unique_labels = np.unique(self.cluster_labels)
        label_to_index = {label: index for index, label in enumerate(unique_labels)}
        self.cluster_labels = np.array([label_to_index[label] for label in self.cluster_labels])
        self.cluster_terms = [self.cluster_terms[label_to_index[label]] for label in self.cluster_labels]

    def merge_similar_clusters2(self, distance='cosine', char_diff_threshold=3, similarity_threshold=0.92):
        logging.info("Merging similar clusters based on distance: {}".format(distance))
        from collections import defaultdict
        merge_mapping = defaultdict(set)

        if distance == 'levenshtein':
            for i, name1 in enumerate(self.cluster_terms):
                for j, name2 in enumerate(self.cluster_terms[i + 1:], start=i + 1):
                    dist = distance(name1, name2)
                    if dist <= char_diff_threshold:
                        merge_mapping[name1].add(name2)
                        logging.info(f"Merging '{name2}' into '{name1}' based on Levenshtein distance")

        elif distance == 'cosine':
            if self.cluster_terms_embeddings is None:
                self.cluster_terms_embeddings = embed_model.encode(self.cluster_terms)
            cos_sim_matrix = pytorch_cos_sim(self.cluster_terms_embeddings, self.cluster_terms_embeddings)
            for i in range(len(self.cluster_terms)):
                for j in range(i + 1, len(self.cluster_terms)):
                    if cos_sim_matrix[i][j] > similarity_threshold:
                        merge_mapping[self.cluster_terms[i]].add(self.cluster_terms[j])
                        #st.write(f"Merging cluster '{self.cluster_terms[j]}' into cluster '{self.cluster_terms[i]}'")
                        logging.info(f"Merging cluster '{self.cluster_terms[j]}' into cluster '{self.cluster_terms[i]}'")

        self._update_cluster_terms_and_labels(merge_mapping)

    def _update_cluster_terms_and_labels(self, merge_mapping):
        # Flatten the merge mapping to a simple name change mapping
        name_change_mapping = {old: new for new, olds in merge_mapping.items() for old in olds}
        # Update cluster terms and labels
        unique_new_terms = list(set(name_change_mapping.values()))
        # replace the old terms with the new terms (name2) otherwise, keep the old terms (name1)
        # self.cluster_terms = [name_change_mapping.get(term, term) for term in self.cluster_terms]
        # self.cluster_labels = np.array([unique_new_terms.index(term) if term in unique_new_terms else term for term in self.cluster_terms])
        self.cluster_terms = [name_change_mapping.get(term, term) for term in self.cluster_terms]
        self.cluster_labels = [unique_new_terms.index(term) if term in unique_new_terms else -1 for term in self.cluster_terms]

        logging.info(f"Total clusters merged: {len(merge_mapping)}")


    def cluster_levenshtein(self, cluster_terms, cluster_labels, char_diff_threshold=3):
        from Levenshtein import distance  # Make sure to import the correct distance function

        merge_map = {}
        # Iterate over term pairs and decide on merging based on the distance
        for idx, term1 in enumerate(cluster_terms):
            for jdx, term2 in enumerate(cluster_terms):
                if idx < jdx and distance(term1, term2) <= char_diff_threshold:  
                    labels_to_merge = [label for label, term_index in enumerate(cluster_labels) if term_index == jdx]
                    for label in labels_to_merge:
                        merge_map[label] = idx  # Map the label to use the term index of term1
                    logging.info(f"Merging '{term2}' into '{term1}'")
                    st.write(f"Merging '{term2}' into '{term1}'")
        # Update the cluster labels
        updated_cluster_labels = [merge_map.get(label, label) for label in cluster_labels]
        # Update string labels to reflect merged labels
        updated_string_labels = [cluster_terms[label] for label in updated_cluster_labels]
        return updated_string_labels

    def cluster_cosine(self, cluster_terms, cluster_labels, similarity_threshold):
        from sklearn.metrics.pairwise import cosine_similarity

        cluster_terms_embeddings = embed_model.encode(cluster_terms)
        # Compute cosine similarity matrix in a vectorized form
        cos_sim_matrix = cosine_similarity(cluster_terms_embeddings, cluster_terms_embeddings)

        merge_map = {}
        n_terms = len(cluster_terms)
        # Iterate only over upper triangular matrix excluding diagonal to avoid redundant computations and self-comparison
        for idx in range(n_terms):
            for jdx in range(idx + 1, n_terms):
                if cos_sim_matrix[idx, jdx] >= similarity_threshold:
                    labels_to_merge = [label for label, term_index in enumerate(cluster_labels) if term_index == jdx]
                    for label in labels_to_merge:
                        merge_map[label] = idx
                    st.write(f"Merging '{cluster_terms[jdx]}' into '{cluster_terms[idx]}'")
                    logging.info(f"Merging '{cluster_terms[jdx]}' into '{cluster_terms[idx]}'")
        # Update the cluster labels
        updated_cluster_labels = [merge_map.get(label, label) for label in cluster_labels]
        # Update string labels to reflect merged labels
        updated_string_labels = [cluster_terms[label] for label in updated_cluster_labels]
        # make a dataframe with index, cluster label and cluster term
        return updated_string_labels

    def merge_similar_clusters(self, cluster_terms, cluster_labels, distance_type='cosine', char_diff_threshold=3, similarity_threshold=0.92):
        if distance_type == 'levenshtein':
            return self.cluster_levenshtein(cluster_terms, cluster_labels, char_diff_threshold)
        elif distance_type == 'cosine':
            return self.cluster_cosine(cluster_terms, cluster_labels, similarity_threshold)


    def plot_embeddings2(self, title=None):
        assert self.reduced_embeddings is not None, "Dimensionality reduction has not been performed yet."
        assert self.cluster_terms is not None, "Cluster TF-IDF analysis has not been performed yet."

        logging.info("Plotting embeddings with TF-IDF colors")

        fig = go.Figure()

        unique_cluster_terms = np.unique(self.cluster_terms)

        for cluster_term in unique_cluster_terms:
            if cluster_term != 'Noise':
                indices = np.where(np.array(self.cluster_terms) == cluster_term)[0]

                # Plot points in the current cluster
                fig.add_trace(
                    go.Scatter(
                        x=self.reduced_embeddings[indices, 0],
                        y=self.reduced_embeddings[indices, 1],
                        mode='markers',
                        marker=dict(
                            size=5,
                            opacity=0.8,
                        ),
                        name=cluster_term,
                        text=self.data[f'{self.column}'].iloc[indices], 
                        hoverinfo='text',
                    )
                )
            else:
                # Plot noise points differently if needed
                fig.add_trace(
                    go.Scatter(
                        x=self.reduced_embeddings[indices, 0],
                        y=self.reduced_embeddings[indices, 1],
                        mode='markers',
                        marker=dict(
                            size=5,
                            opacity=0.5,
                            color='grey'
                        ),
                        name='Noise',
                        text=[self.data[f'{self.column}'][i] for i in indices],  # Adjusted for potential pandas use
                        hoverinfo='text',
                    )
                )
            # else:
            #     indices = np.where(np.array(self.cluster_terms) == 'Noise')[0]

            #     # Plot noise points
            #     fig.add_trace(
            #         go.Scatter(
            #             x=self.reduced_embeddings[indices, 0],
            #             y=self.reduced_embeddings[indices, 1],
            #             mode='markers',
            #             marker=dict(
            #                 size=5,
            #                 opacity=0.8,
            #             ),
            #             name='Noise',
            #             text=self.data[f'{self.column}'].iloc[indices],
            #             hoverinfo='text',
            #         )
            #     )

        fig.update_layout(title=title, showlegend=True, legend_title_text='Top TF-IDF Terms')
        #return fig
        st.plotly_chart(fig, use_container_width=True)  
       #fig.show()
        #logging.info("Embeddings plotted with TF-IDF colors")

    def plot_embeddings3(self, title=None):
        assert self.reduced_embeddings is not None, "Dimensionality reduction has not been performed yet."
        assert self.cluster_terms is not None, "Cluster TF-IDF analysis has not been performed yet."

        logging.info("Plotting embeddings with TF-IDF colors")

        fig = go.Figure()

        unique_cluster_terms = np.unique(self.cluster_terms)

        terms_order = {term: i for i, term in enumerate(np.unique(self.cluster_terms, return_index=True)[0])}
        #indices = np.argsort([terms_order[term] for term in self.cluster_terms])

        # Handling color assignment, especially for noise
        colors = {term: ('grey' if term == 'Noise' else None) for term in unique_cluster_terms}
        color_map = px.colors.qualitative.Plotly  # Default color map from Plotly Express for consistency

        # Apply a custom color map, handling 'Noise' specifically
        color_idx = 0
        for cluster_term in unique_cluster_terms:
            indices = np.where(np.array(self.cluster_terms) == cluster_term)[0]
            if cluster_term != 'Noise':
                marker_color = color_map[color_idx % len(color_map)]
                color_idx += 1
            else:
                marker_color = 'grey'

            fig.add_trace(
                go.Scatter(
                    x=self.reduced_embeddings[indices, 0],
                    y=self.reduced_embeddings[indices, 1],
                    mode='markers',
                    marker=dict(
                        size=5,
                        opacity=(0.5 if cluster_term == 'Noise' else 0.8),
                        color=marker_color
                    ),
                    name=cluster_term,
                    text=self.data[f'{self.column}'].iloc[indices],
                    hoverinfo='text'
                )
            )
        fig.data = sorted(fig.data, key=lambda trace: terms_order[trace.name])
        fig.update_layout(title=title if title else "Embeddings Visualized", showlegend=True, legend_title_text='Top TF-IDF Terms')
        st.plotly_chart(fig, use_container_width=True)


    def plot_embeddings(self, title=None):
        """
        Plots the reduced dimensionality embeddings with clusters indicated.
        
        Args:
            title (str): The title of the plot.
        """
        # Ensure dimensionality reduction and TF-IDF based cluster naming have been performed
        assert self.reduced_embeddings is not None, "Dimensionality reduction has not been performed yet."
        assert self.cluster_terms is not None, "Cluster TF-IDF analysis has not been performed yet."

        logging.info("Plotting embeddings with TF-IDF colors")
        
        fig = go.Figure()
        
        #for i, term in enumerate(self.cluster_terms):
            # Indices of points in the current cluster
        #unique_cluster_ids = np.unique(self.cluster_labels[self.cluster_labels != -1])  # Exclude noise
        unique_cluster_terms = np.unique(self.cluster_terms)
        unique_cluster_labels = np.unique(self.cluster_labels)
            
        for i, (cluster_id, cluster_terms) in enumerate(zip(unique_cluster_labels, unique_cluster_terms)):
            indices = np.where(self.cluster_labels == cluster_id)[0]
            #indices = np.where(self.cluster_labels == i)[0]
            
            # Plot points in the current cluster
            fig.add_trace(
                go.Scatter(
                    x=self.reduced_embeddings[indices, 0],
                    y=self.reduced_embeddings[indices, 1],
                    mode='markers',
                    marker=dict(
                        #color=i,
                        #colorscale='rainbow',
                        size=5,
                        opacity=0.8,
                    ),
                    name=cluster_terms,
                    text=self.data[f'{self.column}'].iloc[indices],
                    hoverinfo='text',
                )
            )
            
        
        fig.update_layout(title=title, showlegend=True, legend_title_text='Top TF-IDF Terms')
        st.plotly_chart(fig, use_container_width=True)
        logging.info("Embeddings plotted with TF-IDF colors")

    def plot_embeddings4(self, title=None, cluster_terms=None, cluster_labels=None, reduced_embeddings=None, column=None, data=None):
        """
        Plots the reduced dimensionality embeddings with clusters indicated.
        
        Args:
            title (str): The title of the plot.
        """
        # Ensure dimensionality reduction and TF-IDF based cluster naming have been performed
        assert reduced_embeddings is not None, "Dimensionality reduction has not been performed yet."
        assert cluster_terms is not None, "Cluster TF-IDF analysis has not been performed yet."

        logging.info("Plotting embeddings with TF-IDF colors")
        
        fig = go.Figure()
        
        # Determine unique cluster IDs and terms, and ensure consistent color mapping
        unique_cluster_ids = np.unique(cluster_labels)
        unique_cluster_terms = [cluster_terms[i] for i in unique_cluster_ids]#if i != -1]  # Exclude noise by ID

        color_map = px.colors.qualitative.Plotly  # Using Plotly Express's qualitative colors for consistency
        color_idx = 0
        
        # Map each cluster ID to a color
        cluster_colors = {}
        for cid in unique_cluster_ids:
            #if cid != -1:  # Exclude noise
                cluster_colors[cid] = color_map[color_idx % len(color_map)]
                color_idx += 1
            #else:
            #    cluster_colors[cid] = 'grey'  # Noise or outliers in grey

        for cluster_id, cluster_term in zip(unique_cluster_ids, unique_cluster_terms):
            indices = np.where(cluster_labels == cluster_id)[0]
            fig.add_trace(
                go.Scatter(
                    x=reduced_embeddings[indices, 0],
                    y=reduced_embeddings[indices, 1],
                    mode='markers',
                    marker=dict(
                        color=cluster_colors[cluster_id],
                        size=5,
                        opacity=0.8#if cluster_id != -1 else 0.5,
                    ),
                    name=cluster_term,
                    text=data[f'{column}'].iloc[indices],
                    hoverinfo='text',
                )
            )
            
        fig.update_layout(
            title=title if title else "Embeddings Visualized",
            showlegend=True,
            legend_title_text='Top TF-IDF Terms',
            legend=dict(
                traceorder='normal',  # 'normal' or 'reversed'; ensures that traces appear in the order they are added
                itemsizing='constant'
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        logging.info("Embeddings plotted with TF-IDF colors")


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_and_predict(data, analyzers, col_names):
    """
    Performs analysis on the data using provided analyzers and makes predictions on specified columns.

    Args:
        data (pd.DataFrame): The dataset for analysis.
        analyzers (list): A list of UAPAnalyzer instances.
        col_names (list): Column names to be analyzed and predicted.
    """
    new_data = pd.DataFrame()
    for i, (column, analyzer) in enumerate(zip(col_names, analyzers)):
        new_data[f'Analyzer_{column}'] = analyzer.__dict__['cluster_terms']
        logging.info(f"Cluster terms extracted for {column}")

    new_data = new_data.fillna('null').astype('category')
    data_nums = new_data.apply(lambda x: x.cat.codes)

    for col in data_nums.columns:
        try:
            categories = new_data[col].cat.categories
            x_train, x_test, y_train, y_test = train_test_split(data_nums.drop(columns=[col]), data_nums[col], test_size=0.2, random_state=42)
            bst, accuracy, preds = train_xgboost(x_train, y_train, x_test, y_test, len(categories))
            plot_results(new_data, bst, x_test, y_test, preds, categories, accuracy, col)
        except Exception as e:
            logging.error(f"Error processing {col}: {e}")
    return new_data

def train_xgboost(x_train, y_train, x_test, y_test, num_classes):
    """
    Trains an XGBoost model and evaluates its performance.

    Args:
        x_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        x_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.
        num_classes (int): The number of unique classes in the target variable.

    Returns:
        bst (Booster): The trained XGBoost model.
        accuracy (float): The accuracy of the model on the test set.
    """
    dtrain = xgb.DMatrix(x_train, label=y_train, enable_categorical=True)
    dtest = xgb.DMatrix(x_test, label=y_test)

    params = {'objective': 'multi:softmax', 'num_class': num_classes, 'max_depth': 6, 'eta': 0.3}
    num_round = 100
    bst = xgb.train(dtrain=dtrain, params=params, num_boost_round=num_round)
    preds = bst.predict(dtest)
    accuracy = accuracy_score(y_test, preds)

    logging.info(f"XGBoost trained with accuracy: {accuracy:.2f}")
    return bst, accuracy, preds

def plot_results(new_data, bst, x_test, y_test, preds, categories, accuracy, col):
    """
    Plots the feature importance, confusion matrix, and contingency table.

    Args:
        bst (Booster): The trained XGBoost model.
        x_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.
        preds (np.array): Predictions made by the model.
        categories (Index): Category names for the target variable.
        accuracy (float): The accuracy of the model on the test set.
        col (str): The target column name being analyzed and predicted.
    """
    fig, axs = plt.subplots(1, 3, figsize=(25, 5), dpi=300)
    fig.suptitle(f'{col.split(sep=".")[-1]} prediction', fontsize=35)

    plot_importance(bst, ax=axs[0], importance_type='gain', show_values=False)
    conf_matrix = confusion_matrix(y_test, preds)
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=categories, yticklabels=categories, ax=axs[1])
    axs[1].set_title(f'Confusion Matrix\nAccuracy: {accuracy * 100:.2f}%')
    # make axes rotated
    axs[1].set_yticklabels(axs[1].get_yticklabels(), rotation=30, ha='right')
    sorted_features = sorted(bst.get_score(importance_type="gain").items(), key=lambda x: x[1], reverse=True)
    # The most important feature is the first element in the sorted list
    most_important_feature = sorted_features[0][0]
    # Create a contingency table
    contingency_table = pd.crosstab(new_data[col], new_data[most_important_feature])

    # resid pearson is used to calculate the residuals, which 
    table = stats.Table(contingency_table).resid_pearson
    #print(table)
    # Perform the chi-squared test
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    # Print the results
    print(f"Chi-squared test for {col} and {most_important_feature}: p-value = {p}")
    
    sns.heatmap(table, annot=True, cmap='Greens', ax=axs[2])
    # make axis rotated
    axs[2].set_yticklabels(axs[2].get_yticklabels(), rotation=30, ha='right')
    axs[2].set_title(f'Contingency Table between {col.split(sep=".")[-1]} and {most_important_feature.split(sep=".")[-1]}\np-value = {p}')    

    plt.tight_layout()
    #plt.savefig(f"{col}_{accuracy:.2f}_prediction_XGB.jpeg", dpi=300)
    return plt

def cramers_v(confusion_matrix):
    """Calculate Cramer's V statistic for categorical-categorical association."""
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    r_corr = r - ((r-1)**2)/(n-1)
    k_corr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((k_corr-1), (r_corr-1)))

def plot_cramers_v_heatmap(data, significance_level=0.05):
    """Plot heatmap of Cramer's V statistic for each pair of categorical variables in a DataFrame."""
    # Initialize a DataFrame to store Cramer's V values
    cramers_v_df = pd.DataFrame(index=data.columns, columns=data.columns, data=np.nan)
    
    # Compute Cramer's V for each pair of columns
    for col1 in data.columns:
        for col2 in data.columns:
            if col1 != col2:  # Avoid self-comparison
                confusion_matrix = pd.crosstab(data[col1], data[col2])
                chi2, p, dof, expected = chi2_contingency(confusion_matrix)
                # Check if the p-value is less than the significance level
                #if p < significance_level:
                #    cramers_v_df.at[col1, col2] = cramers_v(confusion_matrix)
                # alternatively, you can use the following line to include all pairs
                cramers_v_df.at[col1, col2] = cramers_v(confusion_matrix)
    
    # Plot the heatmap
    plt.figure(figsize=(12, 10), dpi=200)
    mask = np.triu(np.ones_like(cramers_v_df, dtype=bool))  # Mask for the upper triangle
    # make a max and min of the cmap
    sns.heatmap(cramers_v_df, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, mask=mask, square=True)
    plt.title(f"Heatmap of Cramér's V (p < {significance_level})")
    return plt


class UAPVisualizer:
    def __init__(self, data=None):
        pass  # Initialization can be added if needed

    def analyze_and_predict(self, data, analyzers, col_names):
        new_data = pd.DataFrame()
        for i, (column, analyzer) in enumerate(zip(col_names, analyzers)):
            new_data[f'Analyzer_{column}'] = analyzer.__dict__['cluster_terms']
            print(f"Cluster terms extracted for {column}")

        new_data = new_data.fillna('null').astype('category')
        data_nums = new_data.apply(lambda x: x.cat.codes)

        for col in data_nums.columns:
            try:
                categories = new_data[col].cat.categories
                x_train, x_test, y_train, y_test = train_test_split(data_nums.drop(columns=[col]), data_nums[col], test_size=0.2, random_state=42)
                bst, accuracy, preds = self.train_xgboost(x_train, y_train, x_test, y_test, len(categories))
                self.plot_results(new_data, bst, x_test, y_test, preds, categories, accuracy, col)
            except Exception as e:
                print(f"Error processing {col}: {e}")

    def train_xgboost(self, x_train, y_train, x_test, y_test, num_classes):
        dtrain = xgb.DMatrix(x_train, label=y_train, enable_categorical=True)
        dtest = xgb.DMatrix(x_test, label=y_test)

        params = {'objective': 'multi:softmax', 'num_class': num_classes, 'max_depth': 6, 'eta': 0.3}
        num_round = 100
        bst = xgb.train(dtrain=dtrain, params=params, num_boost_round=num_round)
        preds = bst.predict(dtest)
        accuracy = accuracy_score(y_test, preds)

        print(f"XGBoost trained with accuracy: {accuracy:.2f}")
        return bst, accuracy, preds

    def plot_results(self, new_data, bst, x_test, y_test, preds, categories, accuracy, col):
        fig, axs = plt.subplots(1, 3, figsize=(25, 5))
        fig.suptitle(f'{col.split(sep=".")[-1]} prediction', fontsize=35)

        plot_importance(bst, ax=axs[0], importance_type='gain', show_values=False)
        conf_matrix = confusion_matrix(y_test, preds)
        sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=categories, yticklabels=categories, ax=axs[1])
        axs[1].set_title(f'Confusion Matrix\nAccuracy: {accuracy * 100:.2f}%')

        sorted_features = sorted(bst.get_score(importance_type="gain").items(), key=lambda x: x[1], reverse=True)
        most_important_feature = sorted_features[0][0]
        contingency_table = pd.crosstab(new_data[col], new_data[most_important_feature])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        print(f"Chi-squared test for {col} and {most_important_feature}: p-value = {p}")
        
        sns.heatmap(contingency_table, annot=True, cmap='Greens', ax=axs[2])
        axs[2].set_title(f'Contingency Table between {col.split(sep=".")[-1]} and {most_important_feature.split(sep=".")[-1]}\np-value = {p}')    

        plt.tight_layout()
        plt.savefig(f"{col}_{accuracy:.2f}_prediction_XGB.jpeg", dpi=300)
        plt.show()

    @staticmethod
    def cramers_v(confusion_matrix):
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        r_corr = r - ((r-1)**2)/(n-1)
        k_corr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr / min((k_corr-1), (r_corr-1)))

    def plot_cramers_v_heatmap(self, data, significance_level=0.05):
        cramers_v_df = pd.DataFrame(index=data.columns, columns=data.columns, data=np.nan)
        
        for col1 in data.columns:
            for col2 in data.columns:
                if col1 != col2:
                    confusion_matrix = pd.crosstab(data[col1], data[col2])
                    chi2, p, dof, expected = chi2_contingency(confusion_matrix)
                    if p < significance_level:
                        cramers_v_df.at[col1, col2] = UAPVisualizer.cramers_v(confusion_matrix)
        
        plt.figure(figsize=(10, 8)),# facecolor="black")
        mask = np.triu(np.ones_like(cramers_v_df, dtype=bool))
        #sns.set_theme(style="dark", rc={"axes.facecolor": "black", "grid.color": "white", "xtick.color": "white", "ytick.color": "white", "axes.labelcolor": "white", "axes.titlecolor": "white"})
        # ax = sns.heatmap(cramers_v_df, annot=True, fmt=".1f", linewidths=.5, linecolor='white', cmap='coolwarm', annot_kws={"color":"white"}, cbar=True, mask=mask, square=True)
        # Customizing the color of the ticks and labels to white
        # plt.xticks(color='white')
        # plt.yticks(color='white')
        sns.heatmap(cramers_v_df, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, mask=mask, square=True)
        plt.title(f"Heatmap of Cramér's V (p < {significance_level})")
        plt.show()

    
    def plot_treemap(self, df, column, top_n=32):
        # Get the value counts and the top N labels
        value_counts = df[column].value_counts()
        top_labels = value_counts.iloc[:top_n].index
        

        # Use np.where to replace all values not in the top N with 'Other'
        revised_column = f'{column}_revised'
        df[revised_column] = np.where(df[column].isin(top_labels), df[column], 'Other')

        # Get the value counts including the 'Other' category
        sizes = df[revised_column].value_counts().values
        labels = df[revised_column].value_counts().index

        # Get a gradient of colors
        colors = list(mcolors.TABLEAU_COLORS.values())

        # Get % of each category
        percents = sizes / sizes.sum()

        # Prepare labels with percentages
        labels = [f'{label}\n {percent:.1%}' for label, percent in zip(labels, percents)]

        # Plot the treemap
        squarify.plot(sizes=sizes, label=labels, alpha=0.7, pad=True, color=colors, text_kwargs={'fontsize': 10})

        ax = plt.gca()

        # Iterate over text elements and rectangles (patches) in the axes for color adjustment
        for text, rect in zip(ax.texts, ax.patches):
            background_color = rect.get_facecolor()
            r, g, b, _ = mcolors.to_rgba(background_color)
            brightness = np.average([r, g, b])
            text.set_color('white' if brightness < 0.5 else 'black')

            # Adjust font size based on rectangle's area and wrap long text
            coef = 0.8
            font_size = np.sqrt(rect.get_width() * rect.get_height()) * coef
            text.set_fontsize(font_size)
            wrapped_text = textwrap.fill(text.get_text(), width=20)
            text.set_text(wrapped_text)

        plt.axis('off')
        plt.gca().invert_yaxis()
        plt.gcf().set_size_inches(20, 12)
        plt.show()



class UAPParser:
    def __init__(self, api_key, model="gpt-3.5-turbo-0125", col=None, format_long=None):
        os.environ['OPENAI_API_KEY'] = api_key
        self.client = OpenAI()
        self.model = model
        self.responses = {}
        self.col = None

    def fetch_response(self, description, format_long):
        INITIAL_WAIT_TIME = 5
        MAX_WAIT_TIME = 600
        MAX_RETRIES = 10

        wait_time = INITIAL_WAIT_TIME
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant which is tasked to assign a trustworthiness value between 0 and 100 to the given first-hand report."},
                        {"role": "user", "content": f'Input report: {description}\n\n Parse data following this json structure; leave missing data empty: {format_long}  Output:'}
                    ]
                )
                return response
            except HTTPError as e:
                if 'TooManyRequests' in str(e):
                    time.sleep(wait_time)
                    wait_time = min(wait_time * 2, MAX_WAIT_TIME)  # Exponential backoff
                else:
                    raise
            except Exception as e:
                print(f"Unexpected error: {e}")
                break

        return None  # Return None if all retries fail

    def process_descriptions(self, descriptions, format_long, max_workers=32):
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_desc = {executor.submit(self.fetch_response, desc, format_long): desc for desc in descriptions}

            for future in stqdm(concurrent.futures.as_completed(future_to_desc), total=len(descriptions)):
                desc = future_to_desc[future]
                try:
                    response = future.result()
                    response_text = response.choices[0].message.content if response else None
                    if response_text:
                        self.responses[desc] = response_text
                except Exception as exc:
                    print(f'Error occurred for description {desc}: {exc}')

    def parse_responses(self):
        parsed_responses = {}
        not_parsed = 0
        try:
            for k, v in self.responses.items():
                try:
                    parsed_responses[k] = json.loads(v)
                except:
                    try:
                        parsed_responses[k] = json.loads(v.replace("'", '"'))
                    except:
                        not_parsed += 1
        except Exception as e:
            print(f"Error parsing responses: {e}")
 
        print(f"Number of unparsed responses: {not_parsed}")
        print(f"Number of parsed responses: {len(parsed_responses)}")
        return parsed_responses

    def responses_to_df(self, col, parsed_responses):
        parsed_df = pd.DataFrame(parsed_responses).T
        parsed_df2 = pd.json_normalize(parsed_df[col])
        parsed_df2.index = parsed_df.index
        return parsed_df2



