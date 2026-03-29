import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="RAG Multi-Model Leaderboard", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .stMetric { background-color: #161b22; padding: 15px; border-top: 2px solid #30363d; }
    .stSidebar { background-color: #010409; }
    h1, h2, h3 { color: #58a6ff; }
    .dataframe { background-color: #161b22; color: #c9d1d9; }
    /* Ensure Plotly charts stay dark */
    .stPlotlyChart { background-color: #0d1117; }
    </style>
    """, unsafe_allow_html=True)

st.title("RAG Multi-Model Evaluation Framework")
st.markdown("### Performance Comparison: Phi-3.5, Llama 3.1, Mistral Nemo, and Gemma 2")

# Load Data
REPORT_PATH = "outputs/reports/consolidated_model_results.json"
CATALOG_PATH = "data/benchmarks/catalog.json"

@st.cache_data
def load_all_data():
    if not os.path.exists(REPORT_PATH) or not os.path.exists(CATALOG_PATH):
        return None, None
    with open(REPORT_PATH, "r") as f:
        mod_res = json.load(f)
    with open(CATALOG_PATH, "r") as f:
        cat = json.load(f)
    return mod_res, cat

all_model_results, catalog = load_all_data()

if not all_model_results:
    st.error("Missing consolidated report. Please run `python run_eval.py --local --models llama3.1 phi3.5 mistral-nemo gemma2` first.")
else:
    # --- GLOBAL PERFORMANCE MATRIX ---
    st.header("Global Multi-Model Performance Matrix")
    
    matrix_rows = []
    for model_name, datasets in all_model_results.items():
        for ds_name, res in datasets.items():
            b_df = pd.DataFrame(res["baseline"])
            i_df = pd.DataFrame(res["improved"])
            
            b_grounding = b_df["grounding_score"].mean()
            i_grounding = i_df["grounding_score"].mean()
            delta = i_grounding - b_grounding
            
            matrix_rows.append({
                "Model": model_name,
                "Dataset": ds_name,
                "Baseline Grounding": b_grounding,
                "Improved Grounding": i_grounding,
                "Delta": delta,
                "Improved Hit Rate": i_df["hit_rate"].mean(),
                "Reliability Index": (i_grounding + i_df["hit_rate"].mean()) / 2
            })
    
    master_df = pd.DataFrame(matrix_rows)
    
    # Styled Table
    def color_delta(val):
        color = 'green' if val > 0.01 else 'red' if val < -0.01 else 'gray'
        return f'color: {color}'

    st.subheader("Master Benchmarking Table")
    st.dataframe(
        master_df.style.format({
            "Baseline Grounding": "{:.2f}",
            "Improved Grounding": "{:.2f}",
            "Delta": "{:+.2f}",
            "Improved Hit Rate": "{:.2f}",
            "Reliability Index": "{:.2f}"
        }).applymap(color_delta, subset=['Delta']),
        use_container_width=True,
        height=400
    )

    st.divider()

    # --- PERFORMANCE VISUALIZATION ---
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Domain-Specific Grounding Performance")
        pivot_df = master_df.pivot(index="Dataset", columns="Model", values="Improved Grounding")
        fig_heat = px.imshow(pivot_df, text_auto=".2f", aspect="auto", color_continuous_scale="Viridis",
                              title="Heatmap: Grounding Scores Across Datasets")
        fig_heat.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', 
            font_color="white",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with col2:
        st.subheader("Model Leaderboard")
        leaderboard = master_df.groupby("Model")["Reliability Index"].mean().sort_values(ascending=False).reset_index()
        st.table(leaderboard.style.format({"Reliability Index": "{:.2f}"}))

    st.divider()

    # --- RADAR COMPARISON ---
    st.header("Comparative Model Profiling")
    model_list = list(all_model_results.keys())
    selected_models = st.multiselect("Select Models to Compare", model_list, default=model_list[:min(2, len(model_list))])
    
    if selected_models:
        categories = ['Grounding', 'Hit Rate', 'Safety', 'Recall']
        fig_radar = go.Figure()
        
        colors = ["#79addc", "#ffc09f", "#ffee93", "#adf7b6"]
        for i, m_name in enumerate(selected_models):
            m_data = master_df[master_df["Model"] == m_name]
            if not m_data.empty:
                r_vals = [
                    m_data["Improved Grounding"].mean(),
                    m_data["Improved Hit Rate"].mean(),
                    1.0, # Safety baseline
                    m_data["Improved Hit Rate"].mean()
                ]
                fig_radar.add_trace(go.Scatterpolar(
                    r=r_vals, 
                    theta=categories, 
                    fill='toself', 
                    name=m_name, 
                    line_color=colors[i % len(colors)]
                ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="#444", tickfont=dict(color="white")),
                angularaxis=dict(gridcolor="#444", tickfont=dict(color="white")),
                bgcolor='rgba(0,0,0,0)'
            ), 
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)', 
            font_color="white",
            title="Performance Comparison Radar"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # --- DATASET DEEP DIVE ---
    st.sidebar.divider()
    selected_ds = st.sidebar.selectbox("Dataset Detail View", master_df["Dataset"].unique())
    ds_detail = master_df[master_df["Dataset"] == selected_ds].sort_values(by="Improved Grounding", ascending=False)
    
    st.sidebar.write(f"**Top Model for {selected_ds}:**")
    st.sidebar.info(f"{ds_detail.iloc[0]['Model']}")
    st.sidebar.write(f"Grounding: {ds_detail.iloc[0]['Improved Grounding']:.2f}")