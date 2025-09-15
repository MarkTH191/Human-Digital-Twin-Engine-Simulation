from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go


def _organ_positions():
    # Rough 3D coordinates for a few organs for demo purposes
    return {
        "Brain": (0, 0, 170),
        "Heart": (0, 0, 120),
        "Liver": (5, 30, 100),
    }


def render_3d_view(sim) -> None:
    st.markdown("""
    <div class='panel'>
      <h3 class='accent'>3D Viewer</h3>
    </div>
    """, unsafe_allow_html=True)

    pos = _organ_positions()
    x, y, z, text, color = [], [], [], [], []
    for organ_node, health in sim.organ_sim.organ_health.items():
        label = sim.graph.nodes[organ_node].get("label", organ_node)
        cx, cy, cz = pos.get(label, (0, 0, 0))
        x.append(cx)
        y.append(cy)
        z.append(cz)
        text.append(f"{label}<br>Health: {health:.2f}")
        # Map health to color intensity
        t = max(0.0, min(1.0, health / 10.0))
        color.append(f"rgba({int((1-t)*255)}, {int(t*255)}, 150, 0.9)")

    scatter = go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers+text",
        text=[t.split('<br>')[0] for t in text],
        textposition="top center",
        marker=dict(size=12, color=color, line=dict(width=1, color="#ffffff")),
        hovertext=text,
        hoverinfo="text",
    )

    fig = go.Figure(data=[scatter])
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="#0b0f14",
        ),
        paper_bgcolor="#0b0f14",
        plot_bgcolor="#0b0f14",
        margin=dict(l=0, r=0, t=0, b=0),
    )

    st.plotly_chart(fig, use_container_width=True)

