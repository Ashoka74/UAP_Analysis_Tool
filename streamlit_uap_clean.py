import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from uap_analyzer import UAPParser, UAPAnalyzer, UAPVisualizer
from Levenshtein import distance
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from stqdm import stqdm
stqdm.pandas()
import streamlit.components.v1 as components
from dateutil import parser
from sentence_transformers import SentenceTransformer
import torch
st.set_option('deprecation.showPyplotGlobalUse', False)


from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)



def load_data(file_path, key='df'):
    return pd.read_hdf(file_path, key=key)


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
    query_model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
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
            if is_categorical_dtype(df_[column]) :# or (df_[column].nunique() < 200 and not is_datetime64_any_dtype(df_[column]) and not is_numeric_dtype(df_[column])):
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
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
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

    # Parsed data
    parsed_tickbox = st.checkbox('Parsed Data')
    if parsed_tickbox:
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
    if st.session_state.stage > 0:
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
        parsed2 = st.session_state.get('new_data', pd.DataFrame())
        parsed2 = filter_dataframe(parsed2)
        col1, col2 = st.columns(2)
        st.dataframe(parsed2)
        with col1:
            col_parsed2 = st.selectbox("Which column do you want to query?", parsed2.columns)
        with col2:
            GEMINI_KEY = st.text_input('Gemini API Key', GEMINI_KEY, type='password', help="Enter your Gemini API key")
        if col_parsed and GEMINI_KEY:
            selected_column_data2 = parsed2[col_parsed2].tolist()
            question2 = st.text_input("Ask a question or leave empty for summarization")
            if st.button("Generate Query") and selected_column_data2:
                st.write(gemini_query(question2, selected_column_data2, GEMINI_KEY))


if __name__ == '__main__':
    main()

#streamlit run streamlit_uap_clean.py --server.enableXsrfProtection=false --theme.primaryColor=#FFA500 --theme.base=dark