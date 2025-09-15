from __future__ import annotations

import streamlit as st


def render_controls(sim) -> None:
    st.markdown("""
    <div class='panel'>
      <h3 class='accent'>Controls</h3>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Time Controls", expanded=True):
        scale = st.slider("Time acceleration (x)", min_value=0.0, max_value=1000.0, value=float(sim.time_scale), step=1.0)
        if scale != sim.time_scale:
            sim.set_time_scale(scale)
        st.write(f"Current time: {sim.get_time_readable()}")
        if st.button("Step +1h"):
            sim.step(1.0)

    with st.expander("Gene Editing", expanded=True):
        # Selections
        organs = sorted({label for label in sim.cell_sim.list_organs()})
        if organs:
            organ = st.selectbox("Organ", organs)
            tissues = sorted(set(sim.cell_sim.list_tissues(organ)))
        else:
            organ = "Liver"
            tissues = []
        tissue = st.selectbox("Tissue", tissues) if tissues else "Hepatic Tissue"
        cell_types = [c for _, _, c in sim.cell_sim.list_cell_types(organ, tissue)]
        cell_label = st.selectbox("Cell Type", cell_types) if cell_types else "Hepatocyte"
        gene = st.text_input("Gene", value="TP53")
        delta = st.slider("Delta", -5.0, 5.0, 1.0, 0.1)
        if st.button("Apply Gene Edit"):
            event = sim.apply_gene_edit(organ, tissue, cell_label, gene, delta)
            st.success(f"Edited {event['gene']} in {cell_label}: {event['before']:.2f} -> {event['after']:.2f}")

    with st.expander("Drug Application", expanded=False):
        uniprot = st.text_input("Target UniProt ID", value="P04637")
        effect = st.slider("Effect Strength", -5.0, 5.0, 1.5, 0.1)
        if st.button("Apply Drug"):
            event = sim.apply_drug(uniprot, effect)
            st.info(f"Applied drug to {event['target']} with effect {event['effect_strength']}")

