from __future__ import annotations

from typing import Dict

import streamlit as st
import pandas as pd


def render_dashboard(sim) -> None:
    st.markdown("""
    <div class='panel'>
      <h3 class='accent'>Dashboard</h3>
    </div>
    """, unsafe_allow_html=True)

    st.caption("System Time")
    st.code(sim.get_time_readable())

    st.caption("Organ Health")
    organ_health: Dict[str, float] = sim.get_organ_health()
    if organ_health:
        df = pd.DataFrame({"Organ": list(organ_health.keys()), "Health": list(organ_health.values())})
        st.bar_chart(df.set_index("Organ"))
    else:
        st.info("Run the simulation step to populate organ metrics.")

    st.caption("Protein Activity (top 10)")
    prot = sim.get_protein_activity()
    if prot:
        items = sorted(prot.items(), key=lambda kv: kv[1], reverse=True)[:10]
        st.dataframe(pd.DataFrame(items, columns=["Protein", "Activity"]))
    else:
        st.write("-")

