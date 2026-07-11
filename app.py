import streamlit as st

st.set_page_config(
   page_title="UAP Analytics",
   page_icon="🛸",
   layout="wide",
   initial_sidebar_state="expanded",
)

from PIL import Image
import base64

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


# if st.toggle('Set background image', True):
#     set_png_as_page_bg('saucer.webp')  # Replace with your background image path

# Global pipeline option — read by parsing.py to optionally post-process
# parsed UAP data through the SCU v2 normalizer.
st.sidebar.toggle(
    'Apply SCU normalization to parsed data',
    value=False,
    key='scu_normalize_enabled',
    help=(
        'When enabled, the UAP Feature Extraction page runs the SCU v2 '
        'normalizer on parsed results — canonicalising countries, states, '
        'witness roles and craft fields, and deriving the SCU five-criterion '
        'eligibility gate. Adds a normalized CSV and audit report to download.'
    ),
)

pg = st.navigation([
            st.Page("preprocessing.py", title="Document Preprocessing (Scrape → OCR → Reports → Table)", icon="🧪"),
            st.Page("rag_search.py", title="Smart-Search (Retrieval Augmented Generations)", icon="🔍"),
            st.Page("deduplication.py", title="Deduplication Studio (Harrier Semantic & Spatial Screening)", icon="🧬"),
            st.Page("parsing.py", title="UAP Feature Extraction (Shape, Speed, Color)", icon="📄"),
            st.Page("analyzing.py", title="Statistical Analysis (UMAP+HDBSCAN, XGBoost, V-Cramer)", icon="🧠"),
            st.Page("magnetic.py", title="Magnetic Anomaly Detection (InterMagnet Stations)", icon="🧲"),
            st.Page("map.py", title="Interactive Map (Tracking variations, Proximity with Military Bases, Nuclear Facilities)", icon="🗺️"),
            st.Page("pdf_ocr.py", title="PDF OCR — Table Extractor (LightOnOCR)", icon="📑"),
        ])
pg.run()
