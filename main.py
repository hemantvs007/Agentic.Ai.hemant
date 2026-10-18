"""
main.py - Autonomous Data Analysis Agent Web Application
Professional, light-themed enterprise UI built with Streamlit and powered by Gemini AI.
"""

import os
import io
import time
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Import helper and orchestrator modules
import helper
import orchestrator

# Load environment variables if available
load_dotenv()

# ==============================================================================
# 1. STREAMLIT PAGE CONFIGURATION & PROFESSIONAL LIGHT STYLING
# ==============================================================================

st.set_page_config(
    page_title=" Data Analyst | Autonomous Agent",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Light Theme CSS (No Gradients, No Emojis)
st.markdown("""
<style>
    /* Typography & Global Reset */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #1E293B;
    }
    
    /* Header Container */
    .app-header {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-top: 4px solid #2563EB;
        padding: 1.5rem 2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
    }
    
    .app-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin: 0;
    }
    
    .app-subtitle {
        font-size: 0.95rem;
        font-weight: 400;
        color: #64748B;
        margin-top: 0.35rem;
    }
    
    .status-badge {
        display: inline-block;
        background-color: #F1F5F9;
        color: #475569;
        border: 1px solid #E2E8F0;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .status-badge-success {
        background-color: #ECFDF5;
        color: #065F46;
        border-color: #A7F3D0;
    }

    .status-badge-primary {
        background-color: #EFF6FF;
        color: #1D4ED8;
        border-color: #BFDBFE;
    }
    
    /* Metric Cards */
    .metric-panel {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem 1.25rem;
    }
    
    .metric-panel-title {
        font-size: 0.78rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.35rem;
    }
    
    .metric-panel-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
    }
    
    .metric-panel-sub {
        font-size: 0.8rem;
        font-weight: 500;
        color: #64748B;
        margin-top: 0.25rem;
    }

    /* Solid Professional Buttons */
    .stButton > button {
        background-color: #2563EB;
        color: #FFFFFF;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 0.55rem 1.25rem;
        border-radius: 6px;
        border: 1px solid #1D4ED8;
        box-shadow: none;
        transition: background-color 0.15s ease-in-out;
    }
    
    .stButton > button:hover {
        background-color: #1D4ED8;
        color: #FFFFFF;
        border-color: #1E40AF;
        box-shadow: none;
    }

    .stButton > button:active {
        background-color: #1E40AF;
    }

    /* Content Cards */
    .section-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
    }
    
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #0F172A;
        margin-bottom: 0.75rem;
    }
    
    .section-desc {
        font-size: 0.88rem;
        color: #64748B;
        margin-bottom: 1rem;
        line-height: 1.4;
    }

    /* Result & Response Container */
    .analysis-output {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #2563EB;
        border-radius: 6px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    
    /* Cleaning Log Item */
    .clean-log-item {
        background-color: #F8FAFC;
        color: #334155;
        border: 1px solid #E2E8F0;
        border-left: 3px solid #10B981;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 6px;
    }

    /* Clean Empty State */
    .empty-state-box {
        background-color: #FFFFFF;
        border: 1px dashed #CBD5E1;
        border-radius: 8px;
        padding: 3rem 2rem;
        text-align: center;
        margin-top: 1rem;
    }

    .empty-state-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #0F172A;
        margin-bottom: 0.5rem;
    }

    .empty-state-desc {
        font-size: 0.9rem;
        color: #64748B;
        max-width: 520px;
        margin: 0 auto 1.25rem auto;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. SESSION STATE MANAGEMENT
# ==============================================================================

if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "df" not in st.session_state:
    st.session_state.df = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None
if "cleaning_logs" not in st.session_state:
    st.session_state.cleaning_logs = []
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "api_key_valid" not in st.session_state:
    st.session_state.api_key_valid = False


# ==============================================================================
# 3. SIDEBAR CONFIGURATION
# ==============================================================================

with st.sidebar:
    st.markdown("### System Configuration")
    st.caption("Autonomous Data Analyst Control Panel")

    # API Key Input
    default_key = os.environ.get("GEMINI_API_KEY", "")
    api_key_input = st.text_input(
        "Gemini API Key",
        value=default_key,
        type="password",
        help="Enter your Google Gemini API Key. Alternatively, set GEMINI_API_KEY in .env.",
        placeholder="Enter API key..."
    )
    
    # Key Status Indicator
    if api_key_input:
        is_valid = orchestrator.initialize_gemini(api_key_input)
        if is_valid:
            st.markdown("""
                <div style="background-color: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 6px; padding: 6px 10px; color: #065F46; font-size: 0.82rem; font-weight: 500; margin-bottom: 0.75rem;">
                    API Key Configured
                </div>
            """, unsafe_allow_html=True)
            st.session_state.api_key_valid = True
        else:
            st.markdown("""
                <div style="background-color: #FEF2F2; border: 1px solid #FECACA; border-radius: 6px; padding: 6px 10px; color: #991B1B; font-size: 0.82rem; font-weight: 500; margin-bottom: 0.75rem;">
                    Configuration Error: Invalid Key
                </div>
            """, unsafe_allow_html=True)
            st.session_state.api_key_valid = False
    else:
        st.markdown("""
            <div style="background-color: #FFFBEB; border: 1px solid #FDE68A; border-radius: 6px; padding: 6px 10px; color: #92400E; font-size: 0.82rem; font-weight: 500; margin-bottom: 0.75rem;">
                Action Required: Enter Gemini API Key
            </div>
        """, unsafe_allow_html=True)
        st.session_state.api_key_valid = False

    st.markdown("---")

    # Model Parameters
    st.markdown("### Model Settings")
    model_keys = orchestrator.get_available_models()
    model_option = st.selectbox(
        "Model Selection",
        options=model_keys,
        index=0,
        format_func=lambda m: orchestrator.AVAILABLE_MODELS.get(m, {}).get("display_name", m),
        help="Select the Gemini architecture for data reasoning and code execution."
    )
    
    temperature = st.slider(
        "Temperature (Determinism)",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        help="Lower values produce more deterministic, factual analytical responses."
    )

    st.markdown("---")

    # Cleaning Options
    st.markdown("### Data Pipeline Controls")
    opt_clean_cols = st.checkbox("Standardize Column Names", value=True, help="Converts headers to clean snake_case.")
    opt_remove_dups = st.checkbox("Deduplicate Records", value=True, help="Removes duplicate rows.")
    opt_missing_mode = st.selectbox(
        "Missing Data Strategy",
        options=["Auto-Impute (Mean/Median/Mode)", "Drop Incomplete Rows", "Keep As Is"],
        index=0
    )
    opt_convert_dates = st.checkbox("Auto-Detect Date Columns", value=True, help="Parses date strings to datetime.")
    opt_clip_outliers = st.checkbox("Clip Extreme Outliers (IQR)", value=False, help="Bounds numerical outliers via 1.5 * IQR.")

    missing_param_map = {
        "Auto-Impute (Mean/Median/Mode)": "auto",
        "Drop Incomplete Rows": "drop",
        "Keep As Is": "none"
    }

    if st.button("Apply Data Pipeline", use_container_width=True):
        if st.session_state.raw_df is not None:
            with st.spinner("Processing dataset..."):
                cleaned_df, logs = helper.clean_dataset(
                    st.session_state.raw_df,
                    clean_column_names=opt_clean_cols,
                    remove_duplicates=opt_remove_dups,
                    handle_missing=missing_param_map[opt_missing_mode],
                    convert_dates=opt_convert_dates,
                    clip_outliers=opt_clip_outliers
                )
                st.session_state.df = cleaned_df
                st.session_state.cleaning_logs = logs
                st.success("Pipeline executed successfully.")
                st.rerun()

    st.markdown("---")

    if st.button("Reset Session", use_container_width=True):
        st.session_state.raw_df = None
        st.session_state.df = None
        st.session_state.dataset_name = None
        st.session_state.cleaning_logs = []
        st.session_state.analysis_history = []
        st.session_state.last_result = None
        st.rerun()


# ==============================================================================
# 4. APPLICATION HEADER
# ==============================================================================

st.markdown("""
<div class="app-header">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
        <div>
            <div class="status-badge status-badge-primary">Autonomous Analytics Platform</div>
            <h1 class="app-title" style="margin-top: 6px;"> Data Analyst</h1>
            <div class="app-subtitle">
                Automated dataset profiling, statistical exploration, interactive visualization, and executive reporting.
            </div>
        </div>
        <div style="text-align: right;">
            <div class="status-badge">Engine: Google Gemini</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# 5. DATA INGESTION SECTION
# ==============================================================================

col_upload, col_sample = st.columns([2.5, 1], gap="medium")

with col_upload:
    st.markdown("##### Upload Dataset")
    uploaded_file = st.file_uploader(
        "Upload dataset file",
        type=["csv", "xlsx", "xls", "json", "parquet", "tsv"],
        help="Select a structured tabular data file (CSV, Excel, JSON, Parquet).",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None and (st.session_state.dataset_name != uploaded_file.name):
        with st.spinner("Ingesting and processing dataset..."):
            raw_df, err = helper.load_data(uploaded_file)
            if err:
                st.error(f"Failed to load file: {err}")
            elif raw_df is not None:
                st.session_state.raw_df = raw_df
                st.session_state.dataset_name = uploaded_file.name
                
                cleaned_df, logs = helper.clean_dataset(
                    raw_df,
                    clean_column_names=opt_clean_cols,
                    remove_duplicates=opt_remove_dups,
                    handle_missing=missing_param_map[opt_missing_mode],
                    convert_dates=opt_convert_dates,
                    clip_outliers=opt_clip_outliers
                )
                st.session_state.df = cleaned_df
                st.session_state.cleaning_logs = logs
                st.session_state.last_result = None
                st.rerun()

with col_sample:
    st.markdown("##### Or Select Benchmark Sample")
    sample_choice = st.selectbox(
        "Benchmark dataset",
        options=["Select a sample...", "E-Commerce Sales", "Customer Churn", "Employee Performance & HR"],
        label_visibility="collapsed"
    )
    
    if sample_choice != "Select a sample..." and (st.session_state.dataset_name != f"Sample: {sample_choice}"):
        with st.spinner(f"Loading {sample_choice} benchmark..."):
            raw_df = helper.get_sample_dataset(sample_choice)
            st.session_state.raw_df = raw_df
            st.session_state.dataset_name = f"Sample: {sample_choice}"
            
            cleaned_df, logs = helper.clean_dataset(
                raw_df,
                clean_column_names=opt_clean_cols,
                remove_duplicates=opt_remove_dups,
                handle_missing=missing_param_map[opt_missing_mode],
                convert_dates=opt_convert_dates,
                clip_outliers=opt_clip_outliers
            )
            st.session_state.df = cleaned_df
            st.session_state.cleaning_logs = logs
            st.session_state.last_result = None
            st.rerun()


# ==============================================================================
# 6. MAIN WORKSPACE
# ==============================================================================

if st.session_state.df is None:
    st.markdown("""
        <div class="empty-state-box">
            <div class="empty-state-title">No Dataset Currently Loaded</div>
            <div class="empty-state-desc">
                Upload a structured file (CSV, Excel, JSON, Parquet) above or choose a benchmark sample to activate the analysis engine.
            </div>
            <div style="display: flex; justify-content: center; gap: 8px; flex-wrap: wrap;">
                <span class="status-badge">Automated Data Cleaning</span>
                <span class="status-badge">Interactive Visualizations</span>
                <span class="status-badge">Executive Summaries</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

else:
    df = st.session_state.df
    summary = helper.get_dataset_summary(df)

    # --------------------------------------------------------------------------
    # METRIC OVERVIEW CARDS
    # --------------------------------------------------------------------------
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
            <div class="metric-panel">
                <div class="metric-panel-title">Total Records</div>
                <div class="metric-panel-value">{summary['num_rows']:,}</div>
                <div class="metric-panel-sub">Active Dataset Rows</div>
            </div>
        """, unsafe_allow_html=True)
    
    with m2:
        st.markdown(f"""
            <div class="metric-panel">
                <div class="metric-panel-title">Features (Columns)</div>
                <div class="metric-panel-value">{summary['num_cols']}</div>
                <div class="metric-panel-sub">{len(summary['numeric_cols'])} Numeric | {len(summary['categorical_cols'])} Categorical</div>
            </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
            <div class="metric-panel">
                <div class="metric-panel-title">Missing Values</div>
                <div class="metric-panel-value">{summary['missing_pct']}%</div>
                <div class="metric-panel-sub">{summary['total_missing']} Total Null Cells</div>
            </div>
        """, unsafe_allow_html=True)

    with m4:
        quality_score = max(0, 100 - int(summary['missing_pct'] * 2) - int(summary['duplicate_pct'] * 3))
        st.markdown(f"""
            <div class="metric-panel">
                <div class="metric-panel-title">Data Health Score</div>
                <div class="metric-panel-value">{quality_score} / 100</div>
                <div class="metric-panel-sub">{summary['duplicates']} Duplicate Records</div>
            </div>
        """, unsafe_allow_html=True)

    st.write("")

    # --------------------------------------------------------------------------
    # TABS: QUERY & ANALYSIS | AUTONOMOUS REPORTS | PROFILING & CLEANING
    # --------------------------------------------------------------------------
    tab_query, tab_auto, tab_data = st.tabs([
        "Query & Analysis",
        "Autonomous Reports",
        "Data Profiling & Inspection"
    ])

    # ==========================================================================
    # TAB 1: QUERY & ANALYSIS
    # ==========================================================================
    with tab_query:
        st.markdown("#### Natural Language Dataset Query")
        st.caption("Ask questions, formulate hypotheses, or request specific statistical breakdowns.")

        # Suggested Queries
        st.markdown("<p style='font-size: 0.8rem; font-weight: 600; color: #64748B; text-transform: uppercase; margin-bottom: 4px;'>Suggested Prompts:</p>", unsafe_allow_html=True)
        sug1, sug2, sug3 = st.columns(3)
        
        preset_prompts = [
            "Which category or segment has the highest performance?",
            "Analyze feature correlations and identify primary drivers.",
            "Detect statistical anomalies and outline recommendations."
        ]
        
        selected_prompt = None
        with sug1:
            if st.button(preset_prompts[0], use_container_width=True):
                selected_prompt = preset_prompts[0]
        with sug2:
            if st.button(preset_prompts[1], use_container_width=True):
                selected_prompt = preset_prompts[1]
        with sug3:
            if st.button(preset_prompts[2], use_container_width=True):
                selected_prompt = preset_prompts[2]

        # Query Input
        user_query = st.text_area(
            "Query Input",
            value=selected_prompt if selected_prompt else "",
            height=95,
            placeholder="e.g., 'Examine the distribution of profit across customer segments and generate a bar chart comparing median values.'",
            label_visibility="collapsed"
        )

        btn_col, _ = st.columns([1, 4])
        with btn_col:
            analyze_clicked = st.button("Run Analysis", use_container_width=True)

        if analyze_clicked:
            if not api_key_input:
                st.error("Gemini API Key is required. Please provide a key in the sidebar.")
            elif not user_query.strip():
                st.warning("Please enter a query or select a suggested prompt above.")
            else:
                with st.spinner("Analyzing dataset, running statistical computations, and generating visuals..."):
                    start_time = time.time()
                    result = orchestrator.run_agent_analysis(
                        api_key=api_key_input,
                        model_name=model_option,
                        user_query=user_query,
                        df=df,
                        temperature=temperature
                    )
                    elapsed = round(time.time() - start_time, 2)
                    st.session_state.last_result = result
                    
                    if result["success"]:
                        st.session_state.analysis_history.append({
                            "query": user_query,
                            "result": result,
                            "time": elapsed
                        })

        # Display Result
        if st.session_state.last_result:
            res = st.session_state.last_result
            if res.get("success"):
                st.markdown("---")
                st.markdown("### Analysis Findings & Report")

                # Plotly Chart
                if res.get("figure") is not None:
                    st.markdown("##### Visual Analytics")
                    try:
                        st.plotly_chart(res["figure"], use_container_width=True)
                    except Exception as e:
                        st.warning(f"Chart rendering note: {e}")

                # Textual Report
                if res.get("text_response"):
                    st.markdown("""<div class="analysis-output">""", unsafe_allow_html=True)
                    st.markdown(res["text_response"])
                    st.markdown("""</div>""", unsafe_allow_html=True)

                # Code Expander
                if res.get("code"):
                    with st.expander("Executed Python / Plotly Code"):
                        st.code(res["code"], language="python")
                        if res.get("code_output"):
                            st.markdown("**Console Output:**")
                            st.text(res["code_output"])
                        if res.get("execution_error"):
                            st.warning(f"Execution notice: {res['execution_error']}")

                # Export Report
                report_md = f"# Analysis Report\n\nDataset: {st.session_state.dataset_name}\n\n{res['text_response']}"
                st.download_button(
                    label="Download Report (Markdown)",
                    data=report_md,
                    file_name="analysis_report.md",
                    mime="text/markdown"
                )

            elif res.get("error"):
                st.error(f"Analysis Error: {res['error']}")

    # ==========================================================================
    # TAB 2: AUTONOMOUS 1-CLICK REPORTS
    # ==========================================================================
    with tab_auto:
        st.markdown("#### Autonomous Analytical Routines")
        st.caption("Trigger multi-step autonomous investigations with pre-configured analytical objectives.")

        r1, r2 = st.columns(2)
        with r1:
            st.markdown("""
                <div class="section-card">
                    <div class="section-title">Comprehensive Exploratory Data Analysis</div>
                    <div class="section-desc">
                        Full-spectrum evaluation of feature distributions, target KPIs, variance patterns, and baseline statistical insights.
                    </div>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Generate EDA Report", key="btn_eda", use_container_width=True):
                if not api_key_input:
                    st.error("Gemini API Key required in sidebar.")
                else:
                    with st.spinner("Generating autonomous EDA report..."):
                        res = orchestrator.generate_autonomous_eda(api_key_input, model_option, df)
                        st.session_state.last_result = res
                        st.rerun()

            st.write("")

            st.markdown("""
                <div class="section-card">
                    <div class="section-title">Correlation & Dependency Analysis</div>
                    <div class="section-desc">
                        Evaluates linear and non-linear feature collinearity, dependency structures, and drivers of top performance metrics.
                    </div>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Generate Correlation Matrix", key="btn_trends", use_container_width=True):
                if not api_key_input:
                    st.error("Gemini API Key required in sidebar.")
                else:
                    with st.spinner("Computing correlation matrix and trends..."):
                        res = orchestrator.generate_trends_and_correlations(api_key_input, model_option, df)
                        st.session_state.last_result = res
                        st.rerun()

        with r2:
            st.markdown("""
                <div class="section-card">
                    <div class="section-title">Anomaly & Statistical Risk Detection</div>
                    <div class="section-desc">
                        Identifies data distribution anomalies, tail risks, statistical outliers, and unusual transaction clusters.
                    </div>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Run Anomaly Scan", key="btn_anom", use_container_width=True):
                if not api_key_input:
                    st.error("Gemini API Key required in sidebar.")
                else:
                    with st.spinner("Scanning for statistical anomalies..."):
                        res = orchestrator.generate_anomaly_detection(api_key_input, model_option, df)
                        st.session_state.last_result = res
                        st.rerun()

            st.write("")

            st.markdown("""
                <div class="section-card">
                    <div class="section-title">Strategic Business Recommendations</div>
                    <div class="section-desc">
                        Synthesizes quantitative findings into prioritized, actionable business initiatives with projected impact.
                    </div>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Generate Strategic Action Plan", key="btn_strat", use_container_width=True):
                if not api_key_input:
                    st.error("Gemini API Key required in sidebar.")
                else:
                    with st.spinner("Synthesizing strategic recommendations..."):
                        res = orchestrator.generate_strategic_recommendations(api_key_input, model_option, df)
                        st.session_state.last_result = res
                        st.rerun()

    # ==========================================================================
    # TAB 3: DATA PROFILING & INSPECTION
    # ==========================================================================
    with tab_data:
        st.markdown("#### Dataset Profiling & Applied Pipeline")

        # Applied Pipeline Logs
        if st.session_state.cleaning_logs:
            st.markdown("##### Pipeline Transformations Applied:")
            for log in st.session_state.cleaning_logs:
                st.markdown(f'<div class="clean-log-item">{log}</div>', unsafe_allow_html=True)
            st.write("")

        # Data Table Preview
        st.markdown(f"##### Data Preview ({summary['num_rows']} rows × {summary['num_cols']} columns):")
        st.dataframe(df.head(100), use_container_width=True)

        # Export Cleaned Data
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download Cleaned Data (CSV)",
            data=csv_buffer.getvalue(),
            file_name="cleaned_data.csv",
            mime="text/csv"
        )

        st.markdown("---")

        # Column Schema Table
        st.markdown("##### Column Schema & Distribution Details:")
        if not summary['columns_info'].empty:
            st.dataframe(summary['columns_info'], use_container_width=True)

        # Numerical Descriptive Stats
        if not summary['describe_num'].empty:
            st.markdown("##### Descriptive Statistics (Numerical):")
            st.dataframe(summary['describe_num'], use_container_width=True)


# ==============================================================================
# 7. FOOTER
# ==============================================================================

st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid #E2E8F0; color: #64748B; font-size: 0.82rem;">
    Antigravity Data Analyst • Enterprise Data Intelligence Platform
</div>
""", unsafe_allow_html=True)
