import streamlit as st
from st_paywall import add_auth

add_auth(required=False)

st.write(st.session_state.email)
st.write(st.session_state.user_subscribed)

if "buttons" in st.session_state:
   st.session_state.buttons = st.session_state.buttons

# st.set_page_config(
#     page_title="UAP Analytics",
#     page_icon="🛸",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

pg = st.navigation([
            st.Page("rag_search.py", title="Smart-Search (Retrieval Augmented Generations)", icon="🔍"),
            st.Page("parsing.py", title="UAP Feature Extraction (Shape, Speed, Color)", icon="📄"),
            st.Page("analyzing.py", title="Statistical Analysis (UMAP+HDBSCAN, XGBoost, V-Cramer)", icon="🧠"),
            st.Page("magnetic.py", title="Magnetic Anomaly Detection (InterMagnet Stations)", icon="🧲"),
            st.Page("map.py", title="Interactive Map", icon="🗺️"),
        ])

pg.run()
