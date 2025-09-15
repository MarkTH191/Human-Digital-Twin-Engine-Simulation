import streamlit as st
from ui.dashboard import render_dashboard
from ui.controls import render_controls
from ui.visualizer_3d import render_3d_view
from engine.body_simulator import BodySimulator
from data.preprocess import build_minimal_knowledge_graph

st.set_page_config(page_title="Human Digital Twin", layout="wide")

@st.cache_resource(show_spinner=False)
def get_body_simulator():
    kg = build_minimal_knowledge_graph()
    return BodySimulator(knowledge_graph=kg)


def init_state():
    if "sim" not in st.session_state:
        st.session_state.sim = get_body_simulator()
    if "experiment_log" not in st.session_state:
        st.session_state.experiment_log = []


init_state()

# Dark theme styling
st.markdown(
    """
    <style>
    .stApp { background-color: #0b0f14; color: #e6edf3; }
    .css-18ni7ap, .css-1kyxreq { background: #0b0f14 !important; }
    .stCheckbox, .stSelectbox, .stSlider { color: #e6edf3 !important; }
    .neon { color: #00e5ff; }
    .panel { background: #0f1620; border: 1px solid #1f2937; border-radius: 8px; padding: 1rem; }
    .accent { color: #a78bfa; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("""<h1 class='neon'>Human Digital Twin</h1>""", unsafe_allow_html=True)

left, center, right = st.columns([3, 5, 4])
with left:
    render_controls(st.session_state.sim)
with center:
    render_3d_view(st.session_state.sim)
with right:
    render_dashboard(st.session_state.sim)

# Export control
st.markdown("---")
st.subheader("Export Experiment")
col1, col2 = st.columns([2, 3])
with col1:
    file_name = st.text_input("File name", value="experiment_export.json")
with col2:
    if st.button("Export", type="primary"):
        content = st.session_state.sim.export_experiment()
        st.download_button(
            label="Download JSON",
            data=content,
            file_name=file_name,
            mime="application/json",
        )