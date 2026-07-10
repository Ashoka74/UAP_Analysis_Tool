import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
import torch
from uap_analyzer import get_embed_model
from sentence_transformers.util import pytorch_cos_sim, pairwise_cos_sim
#from stqdm.notebook import stqdm
#stqdm.pandas()
import logging
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap
import fast_hdbscan
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
from tqdm import tqdm
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
import spaces

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
        self.model = None  # lazy-loaded via get_embed_model() on first use
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

    @spaces.GPU
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
        return get_embed_model().encode(data_column.tolist(), show_progress_bar=True)

    @spaces.GPU
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
        
    @spaces.GPU
    def cluster_data(self, method='HDBSCAN', **kwargs):
        """
        Clusters the reduced dimensionality data using the specified clustering method.
        
        Args:
            method (str): The clustering method to use ('HDBSCAN' or 'KMeans').
            **kwargs: Additional keyword arguments for the clustering method.
        """
        logging.info(f"Clustering data using {method}")
        if method == 'HDBSCAN':
            clusterer = fast_hdbscan.HDBSCAN(**kwargs)
        elif method == 'KMeans':
            clusterer = KMeans(**kwargs)
        else:
            raise ValueError("Unsupported clustering method")
        
        clusterer.fit(self.reduced_embeddings)
        self.cluster_labels = clusterer.labels_
        logging.info(f"Data clustering complete using {method}")

    @spaces.GPU
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
            self.cluster_terms_embeddings = get_embed_model().encode(self.cluster_terms)
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
                self.cluster_terms_embeddings = get_embed_model().encode(self.cluster_terms)
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
        
    @spaces.GPU
    def cluster_cosine(self, cluster_terms, cluster_labels, similarity_threshold):
        from sklearn.metrics.pairwise import cosine_similarity

        cluster_terms_embeddings = get_embed_model().encode(cluster_terms)
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

@spaces.GPU
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


@spaces.GPU
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

            for future in tqdm(concurrent.futures.as_completed(future_to_desc), total=len(descriptions)):
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
        if col is not None:
            parsed_df2 = pd.json_normalize(parsed_df[col])
            parsed_df2.index = parsed_df.index
        else:
            parsed_df2 = pd.json_normalize(parsed_df)
            parsed_df2.index = parsed_df.index
        return parsed_df2


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Levenshtein import distance
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import streamlit.components.v1 as components
from dateutil import parser
from sentence_transformers import SentenceTransformer
import torch
# st.set_option('deprecation.showPyplotGlobalUse', False)


from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)



def load_data(file_path, key='df'):
    from data_fetch import load_uap_dataset
    return load_uap_dataset(file_path, key=key)


def gemini_query(question, selected_data, gemini_key):

    if question == "":
        question = "Summarize the following data in relevant bullet points"

    import pathlib
    import textwrap

    import google.generativeai as genai

    from IPython.display import display
    from IPython.display import Markdown


    def to_markdown(text):
        text = text.replace('•', '  *')
        return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))
    
    # selected_data is a list
    # remove empty

    filtered = [str(x) for x in selected_data if str(x) != '' and x is not None]
    # make a string
    context = '\n'.join(filtered)

    genai.configure(api_key=gemini_key)
    query_model = genai.GenerativeModel('models/gemini-3.1-pro-preview')
    response = query_model.generate_content([f"{question}\n Answer based on this context: {context}\n\n"])
    return(response.text)

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    try:
        modify = st.checkbox("Add filters on raw data")
    except:
        try:
            modify = st.checkbox("Add filters on processed data")
        except:
            try:
                modify = st.checkbox("Add filters on parsed data")
            except:
                pass

    if not modify:
        return df

    df_ = df.copy()
    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df_.columns:
        if is_object_dtype(df_[col]):
            try:
                df_[col] = pd.to_datetime(df_[col])
            except Exception:
                try:
                    df_[col] = df_[col].apply(parser.parse)
                except Exception:
                    pass
        if is_datetime64_any_dtype(df_[col]):
            df_[col] = df_[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df_.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 200 unique values as categorical if not date or numeric
            if is_categorical_dtype(df_[column]) or (df_[column].nunique() < 120 and not is_datetime64_any_dtype(df_[column]) and not is_numeric_dtype(df_[column])):
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df_[column].unique(),
                    default=list(df_[column].unique()),
                )
                df_ = df_[df_[column].isin(user_cat_input)]
            elif is_numeric_dtype(df_[column]):
                _min = float(df_[column].min())
                _max = float(df_[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df_ = df_[df_[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df_[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df_[column].min(),
                        df_[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df_ = df_.loc[df_[column].between(start_date, end_date)]
            else:
                try: # To avoid multiple buttons with same ID
                    user_text_input = right.text_input(
                        f"Substring or regex in {column}",
                    )
                except:
                    try:
                         user_text_input = right.text_input(
                        f"Substring or regex {column}",
                        )
                    except Exception as e:
                        print(f'Error : {e}')
                        pass
            
                if user_text_input:
                    df_ = df_[df_[column].astype(str).str.contains(user_text_input)]
        # write len of df after filtering with % of original
        st.write(f"{len(df_)} rows ({len(df_) / len(df) * 100:.2f}%)")
    return df_

def merge_clusters(df, column):
    cluster_terms_ = df.__dict__['cluster_terms']
    cluster_labels_ = df.__dict__['cluster_labels']
    label_name_map = {label: cluster_terms_[label] for label in set(cluster_labels_)}
    merge_map = {}
    # Iterate over term pairs and decide on merging based on the distance
    for idx, term1 in enumerate(cluster_terms_):
        for jdx, term2 in enumerate(cluster_terms_):
            if idx < jdx and distance(term1, term2) <= 3:  # Adjust threshold as needed
                # Decide to merge labels corresponding to jdx into labels corresponding to idx
                # Find labels corresponding to jdx and idx
                labels_to_merge = [label for label, term_index in enumerate(cluster_labels_) if term_index == jdx]
                for label in labels_to_merge:
                    merge_map[label] = idx  # Map the label to use the term index of term1

    # Update the analyzer with the merged numeric labels 
    updated_cluster_labels_ = [merge_map[label] if label in merge_map else label for label in cluster_labels_]

    df.__dict__['cluster_labels'] = updated_cluster_labels_
    # Optional: Update string labels to reflect merged labels
    updated_string_labels = [cluster_terms_[label] for label in updated_cluster_labels_]
    df.__dict__['string_labels'] = updated_string_labels
    return updated_string_labels

def analyze_and_predict(data, analyzers, col_names, clusters):
    visualizer = UAPVisualizer()
    new_data = pd.DataFrame()
    for i, column  in enumerate(col_names):
        #new_data[f'Analyzer_{column}'] = analyzer.__dict__['cluster_labels']
        new_data[f'Analyzer_{column}'] = clusters[column]
        data[f'Analyzer_{column}'] = clusters[column]
        #data[f'Analyzer_{column}'] = analyzer.__dict__['cluster_labels']

        print(f"Cluster terms extracted for {column}")

    for col in data.columns:
        if 'Analyzer' in col:
            data[col] = data[col].astype('category')

    new_data = new_data.fillna('null').astype('category')
    data_nums = new_data.apply(lambda x: x.cat.codes)

    for col in data_nums.columns:
        try:
            categories = new_data[col].cat.categories
            x_train, x_test, y_train, y_test = train_test_split(data_nums.drop(columns=[col]), data_nums[col], test_size=0.2, random_state=42)
            bst, accuracy, preds = visualizer.train_xgboost(x_train, y_train, x_test, y_test, len(categories))
            fig = visualizer.plot_results(new_data, bst, x_test, y_test, preds, categories, accuracy, col)
            with st.status(f"Charts Analyses: {col}", expanded=True) as status:
                st.pyplot(fig)
                status.update(label=f"Chart Processed: {col}", expanded=False)   
        except Exception as e:
            print(f"Error processing {col}: {e}")
            continue
    return new_data, data

def main():
    from config import API_KEY, GEMINI_KEY, FORMAT_LONG

    with torch.no_grad():
        torch.cuda.empty_cache()

    st.set_page_config(
        page_title="UAP ANALYSIS",
        page_icon=":alien:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title('UAP Analysis Dashboard')

    # Initialize session state
    if 'analyzers' not in st.session_state:
        st.session_state['analyzers'] = []
    if 'col_names' not in st.session_state:
        st.session_state['col_names'] = []
    if 'clusters' not in st.session_state:
        st.session_state['clusters'] = {}
    if 'new_data' not in st.session_state:
        st.session_state['new_data'] = pd.DataFrame()
    if 'dataset' not in st.session_state:
        st.session_state['dataset'] = pd.DataFrame()
    if 'data_processed' not in st.session_state:
        st.session_state['data_processed'] = False
    if 'stage' not in st.session_state:
        st.session_state['stage'] = 0
    if 'filtered_data' not in st.session_state:
        st.session_state['filtered_data'] = None

    # Load dataset
    data_path = 'parsed_files_distance_embeds.h5'
    parsed = load_data(data_path).drop(columns=['embeddings']).head(10000)

    # Unparsed data
    unparsed_tickbox = st.checkbox('Unparsed Data')
    if unparsed_tickbox:
        unparsed = st.file_uploader("Upload Raw DataFrame", type=["csv", "xlsx"])
        if unparsed is not None:
            try:
                data = pd.read_csv(unparsed) if unparsed.type == "text/csv" else pd.read_excel(unparsed)
                filtered_data = filter_dataframe(data)
                st.dataframe(filtered_data)
            except Exception as e:
                st.error(f"An error occurred while reading the file: {e}")
            modify_json = st.checkbox('Custom JSON')
            API_KEY = st.text_input('OpenAI API Key', API_KEY, type='password', help="Enter your OpenAI API key")
            if not API_KEY:
                st.warning("Please enter your API key to proceed.")
            if modify_json:
                FORMAT_LONG = st.text_area('Custom JSON', FORMAT_LONG, height=500)
            
            # If the DataFrame is successfully created, allow the user to select a column
            col_unparsed = st.selectbox("Select column corresponding to text", data.columns)
            
            if st.button("Process Column"):
                selected_column_data = filtered_data[col_unparsed].tolist()
                st.write("Column Data:", selected_column_data)
                st.session_state.result = selected_column_data
            if 'parsed_responses' not in st.session_state: # Button to trigger parsing of descriptions
                    with st.status(f"Parsing..", expanded=True) as status:
                        try:
                            st.write("Parsing descriptions...")
                            parser = UAPParser(api_key=API_KEY, model='gpt-3.5-turbo-0125', col=st.session_state.result)
                            #descriptions = unparsed['description'].tolist()
                            descriptions = st.session_state.result
                            format_long = FORMAT_LONG
                            parser.process_descriptions(descriptions, format_long)
                            parsed_responses = parser.parse_responses()
                            try:
                                responses_df = parser.responses_to_df('sightingDetails', parsed_responses)
                            except Exception as e:
                                status.update(label=f"Error parsing: {e}", state="error")
                                responses_df = parser.responses_to_df(None, parsed_responses)
                            st.dataframe(responses_df)
                            st.session_state['parsed_responses'] = responses_df.copy()
                            status.update(label="Parsing complete", expanded=False)
                        except Exception as e:
                            status.update(label=f"Parsing failed : {e}", state="error")
            else:
                # Prompt the user to upload a file if they haven't already
                st.warning("Please upload a file to proceed.")

    # Parsed data
    parsed_tickbox = st.checkbox('Parsed Data')
    if parsed_tickbox:
        if 'parsed_responses' in st.session_state:
            parsed_responses = filter_dataframe(st.session_state['parsed_responses'])
            st.session_state['parsed_responses'] = parsed_responses
        else:
            parsed_responses = filter_dataframe(parsed)
            st.session_state['parsed_responses'] = parsed_responses
        col1, col2 = st.columns(2)
        st.dataframe(parsed_responses)
        with col1:
            col_parsed = st.selectbox("Which column do you want to query?", st.session_state['parsed_responses'].columns)
        with col2:
            GEMINI_KEY = st.text_input('Gemini API Key', GEMINI_KEY, type='password', help="Enter your Gemini API key")
        if col_parsed and GEMINI_KEY:
            selected_column_data = st.session_state['parsed_responses'][col_parsed].tolist()
            question = st.text_input("Ask a question or leave empty for summarization")
            if st.button("Generate Query") and selected_column_data:
                st.write(gemini_query(question, selected_column_data, GEMINI_KEY))
        st.session_state['stage'] = 1

    # Analyze data
    if st.session_state.stage > 0 and st.session_state.stage < 10 and parsed_responses is not None:
        columns_to_analyze = st.multiselect(
            label='Select columns to analyze',
            options=parsed_responses.columns
        )
        if columns_to_analyze:
            analyzers = []
            col_names = []
            clusters = {}
            for column in columns_to_analyze:
                with torch.no_grad():    
                    with st.status(f"Processing {column}", expanded=True) as status:
                        analyzer = UAPAnalyzer(parsed_responses, column)
                        st.write(f"Processing {column}...")
                        analyzer.preprocess_data(top_n=32)
                        st.write("Reducing dimensionality...")
                        analyzer.reduce_dimensionality(method='UMAP', n_components=2, n_neighbors=15, min_dist=0.1)
                        st.write("Clustering data...")
                        analyzer.cluster_data(method='HDBSCAN', min_cluster_size=15)
                        analyzer.get_tf_idf_clusters(top_n=1)
                        st.write("Naming clusters...")
                        analyzers.append(analyzer)
                        col_names.append(column)
                        clusters[column] = analyzer.merge_similar_clusters(cluster_terms=analyzer.__dict__['cluster_terms'], cluster_labels=analyzer.__dict__['cluster_labels'])
                        status.update(label=f"Processing {column} complete", expanded=False)
            st.session_state['analyzers'] = analyzers
            st.session_state['col_names'] = col_names
            st.session_state['clusters'] = clusters
            
            # save space
            parsed = None
            analyzers = None
            col_names = None
            clusters = None

            if st.session_state['clusters'] is not None:
                try:
                    new_data, parsed_responses = analyze_and_predict(parsed_responses, st.session_state['analyzers'], st.session_state['col_names'], st.session_state['clusters'])       
                    st.session_state['dataset'] = parsed_responses
                    st.session_state['new_data'] = new_data
                    st.session_state['data_processed'] = True
                except Exception as e:
                    st.write(f"Error processing data: {e}")
                


            if st.session_state['data_processed']:
                try:
                    visualizer = UAPVisualizer(data=st.session_state['new_data'])
                    #new_data = pd.DataFrame()  # Assuming new_data is prepared earlier in the code
                    fig2 = visualizer.plot_cramers_v_heatmap(data=st.session_state['new_data'], significance_level=0.05)
                    with st.status(f"Cramer's V Chart", expanded=True) as statuss:
                        st.pyplot(fig2)
                        statuss.update(label="Cramer's V chart plotted", expanded=False)   
                except Exception as e:
                    st.write(f"Error plotting Cramers V: {e}")

                for i, column in enumerate(st.session_state['col_names']):
                    #if stateful_button(f"Show {column} clusters {i}", key=f"show_{column}_clusters"):
                    if st.session_state['data_processed']:
                        with st.status(f"Show clusters {column}", expanded=True) as stats:
                            # plot_embeddings4(self, title=None, cluster_terms=None, cluster_labels=None, reduced_embeddings=None, column=None, data=None):
                            fig3 = st.session_state['analyzers'][i].plot_embeddings4(title=f"{column} clusters", cluster_terms=st.session_state['analyzers'][i].__dict__['cluster_terms'], cluster_labels=st.session_state['analyzers'][i].__dict__['cluster_labels'], reduced_embeddings=st.session_state['analyzers'][i].__dict__['reduced_embeddings'], column=f'Analyzer_{column}', data=st.session_state['new_data'])
                            stats.update(label=f"Show clusters {column} complete", expanded=False)

    if st.session_state['data_processed']:
        parsed2 = st.session_state.get('dataset', pd.DataFrame())
        if parsed2 is not None:
            st.session_state['stage'] = 10
        parsed2 = filter_dataframe(parsed2)
        col1, col2 = st.columns(2)
        st.dataframe(parsed2)
        with col1:
            col_parsed2 = st.selectbox("Which columns do you want to query?", parsed2.columns)
        with col2:
            GEMINI_KEY = st.text_input('Gemini API Key', GEMINI_KEY, type='password', help="Enter Gemini API key")
        if col_parsed and GEMINI_KEY:
            selected_column_data2 = parsed2[col_parsed2].tolist()
            question2 = st.text_input("Ask a question / leave empty for summarization")
            if st.button("Generate Query") and selected_column_data2:
                st.write(gemini_query(question2, selected_column_data2, GEMINI_KEY))


if __name__ == '__main__':
    main()

#streamlit run streamlit_uap_clean.py --server.enableXsrfProtection=false --theme.primaryColor=#FFA500 --theme.base=dark