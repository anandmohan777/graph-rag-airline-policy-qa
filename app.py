import time
import asyncio
import io
import csv
import streamlit as st
import pandas as pd
from langchain_ollama import OllamaLLM
from standard_rag import StandardRAG
from graph_rag import GraphRAG
from evaluator import LLMJudgeEvaluator


# ============================================================================
# CONFIGURATION
# ============================================================================
LLM_NAME = "gemma3:27b"

KNOWN_AIRLINES = ['lufthansa', 'britishairways', 'airindia', 'singaporeairlines']

# ============================================================================
# STREAMLIT UI WITH DUAL MODE
# ============================================================================
def main():
    st.set_page_config(
        page_title="RAG vs GraphRAG Comparison",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for modern look
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #666;
            margin-bottom: 2rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding-left: 20px;
            padding-right: 20px;
            background-color: #f0f2f6;
            border-radius: 5px 5px 0 0;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: #667eea;
            color: white;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #667eea;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<h1 class="main-header">🔬 RAG vs GraphRAG Comparison Platform</h1>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">MSc Dissertation: Augmenting Customer Support Chatbots with GraphRAG for Booking Policy Documents</p>',
        unsafe_allow_html=True)

    # Initialize session state
    if 'standard_rag' not in st.session_state:
        st.session_state.standard_rag = None
        st.session_state.graph_rag = None
        st.session_state.both_initialized = False
        st.session_state.judge_evaluator = None

    # Sidebar for system controls
    with st.sidebar:
        st.header("⚙️ System Controls")

        # Initialize systems
        if st.button("🚀 Initialize Both Systems", use_container_width=True, type="primary"):
            with st.spinner("Initializing systems..."):
                # Initialize Standard RAG
                rag = StandardRAG()
                success, msg = rag.initialize()
                if success:
                    st.session_state.standard_rag = rag
                    st.success("✅ Standard RAG ready")
                else:
                    st.error(msg)

                # Initialize GraphRAG
                graphrag = GraphRAG()
                success, msg = graphrag.initialize()
                if success:
                    st.session_state.graph_rag = graphrag
                    st.success("✅ GraphRAG ready")
                else:
                    st.error(msg)

                # Initialize Judge Evaluator
                if st.session_state.standard_rag and st.session_state.graph_rag:
                    st.session_state.judge_evaluator = LLMJudgeEvaluator()
                    st.session_state.judge_evaluator.set_rag_systems(
                        st.session_state.standard_rag,
                        st.session_state.graph_rag
                    )
                    st.session_state.both_initialized = True
                    st.success("✅ All systems operational!")

        st.divider()

        # System status
        st.subheader("📊 System Status")
        col1, col2 = st.columns(2)
        with col1:
            rag_status = "🟢" if st.session_state.standard_rag and st.session_state.standard_rag.initialized else "🔴"
            st.metric("Standard RAG", rag_status)
        with col2:
            graphrag_status = "🟢" if st.session_state.graph_rag and st.session_state.graph_rag.initialized else "🔴"
            st.metric("GraphRAG", graphrag_status)

        st.divider()

        # Index management
        st.subheader("🔧 Index Management")
        if st.button("🔄 Rebuild RAG (ChromaDB)", use_container_width=True):
            if st.session_state.standard_rag:
                with st.spinner("Rebuilding ChromaDB..."):
                    success, msg = st.session_state.standard_rag.rebuild_index()
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
            else:
                st.warning("Initialize Standard RAG first")

        if st.button("🔄 Rebuild GraphRAG (Neo4j Graph)", use_container_width=True):
            if st.session_state.graph_rag:
                with st.spinner("Rebuilding Neo4j graph..."):
                    success, msg = st.session_state.graph_rag.rebuild_graph()
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
            else:
                st.warning("Initialize GraphRAG first")

    # Main content area with tabs
    tab1, tab2 = st.tabs(["🔍 Live Query Comparison", "📊 Batch Evaluation (LLM-as-a-Judge)"])

    # ========================================================================
    # TAB 1: LIVE QUERY COMPARISON (Search functionality)
    # ========================================================================
    with tab1:
        st.markdown("### Compare RAG Systems in Real-Time")
        st.markdown("Ask questions and see how Standard RAG and GraphRAG respond side-by-side.")

        # Query input
        col1, col2 = st.columns([4, 1])
        with col1:
            query = st.text_input(
                "Enter your booking policy question:",
                placeholder="e.g., What are the refund conditions for Economy Basic tickets?",
                key="live_query_input"
            )
        with col2:
            st.markdown("&nbsp;")
            search_button = st.button("🔎 Search", use_container_width=True, type="primary")

        if search_button and query:
            if not st.session_state.both_initialized:
                st.error("⚠️ Please initialize both systems first!")
            else:
                st.markdown("---")

                # Create two columns for parallel display
                left_col, right_col = st.columns(2, gap="large")

                # Execute queries in parallel
                with st.spinner("Querying both systems..."):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    rag_result, graphrag_result = loop.run_until_complete(
                        asyncio.gather(
                            st.session_state.standard_rag.query_async(query),
                            st.session_state.graph_rag.query_async(query)
                        )
                    )
                    loop.close()

                # LEFT COLUMN: Standard RAG
                with left_col:
                    st.markdown("### 📚 Standard RAG")
                    st.caption("Vector similarity retrieval")

                    if rag_result.get("success"):
                        st.info(rag_result["answer"])

                        with st.expander("📄 Retrieved Chunks"):
                            for chunk in rag_result.get("chunks", [])[:4]:
                                st.markdown(f"**{chunk.get('airline', 'Unknown')}** - {chunk.get('section', 'N/A')}")
                                st.caption(chunk.get('content', '')[:200] + "...")
                                st.divider()

                        col1, col2, col3 = st.columns(3)
                        col1.metric("Time", f"{rag_result['response_time']:.2f}s")
                        col2.metric("Chunks", rag_result['num_chunks'])
                        col3.metric("Intent", rag_result.get('intent', 'N/A'))
                    else:
                        st.error(f"Error: {rag_result.get('error', 'Unknown')}")

                # RIGHT COLUMN: GraphRAG
                with right_col:
                    st.markdown("### 🕸️ GraphRAG")
                    st.caption("Vector + graph traversal")

                    if graphrag_result.get("success"):
                        st.success(graphrag_result["answer"])

                        with st.expander("🔗 Graph Reasoning Path"):
                            traversal = graphrag_result["traversal_path"]
                            st.markdown("**Initial Nodes:**")
                            for node in traversal["initial_nodes"]:
                                st.markdown(f"- `{node}`")

                            if traversal["connected_nodes"]:
                                st.markdown("**Connected Nodes:**")
                                for node in traversal["connected_nodes"]:
                                    st.markdown(f"- `{node}` *(via REFERENCES)*")

                            st.metric("Reasoning Hops", traversal['relationships_used'])

                        col1, col2, col3 = st.columns(3)
                        col1.metric("Time", f"{graphrag_result['response_time']:.2f}s")
                        col2.metric("Sections", graphrag_result['num_sections'])
                        col3.metric("Intent", graphrag_result.get('intent', 'N/A'))
                    else:
                        st.error(f"Error: {graphrag_result.get('error', 'Unknown')}")

    # ========================================================================
    # TAB 2: BATCH EVALUATION (LLM-as-a-Judge)
    # ========================================================================
    # ========================================================================
    # TAB 2: BATCH EVALUATION (LLM-as-a-Judge) - APPROACH 1
    # ========================================================================
    with tab2:
        st.info(
            "📋 This will automatically evaluate both Standard RAG and GraphRAG systems using the configured judge LLM.")
        st.markdown("### 📊 LLM-as-a-Judge Batch Evaluation")
        st.markdown(
            "Upload a CSV file with test questions and reference answers for comprehensive automated evaluation.")

        if not st.session_state.both_initialized:
            st.warning("⚠️ Please initialize both systems first before running evaluations.")
            st.stop()

        # Set default values
        judge_model = LLM_NAME

        # File upload section
        st.markdown("#### 📤 Upload Your Evaluation Dataset")
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=["csv"],
            help="Required columns: 'question' and 'golden answer' (or 'reference'/'expected')"
        )

        if uploaded_file is not None:
            try:
                # Read CSV
                content = uploaded_file.read().decode("utf-8", errors="replace")
                reader = csv.DictReader(io.StringIO(content))
                rows = list(reader)

                if not rows:
                    st.warning("No data found in CSV file.")
                    st.stop()

                # Find columns (case-insensitive)
                columns = [col.lower() for col in rows[0].keys()]
                q_col = next((col for col in rows[0].keys() if col.lower() in ['question', 'query']), None)
                r_col = next(
                    (col for col in rows[0].keys() if col.lower() in ['golden answer', 'reference', 'expected']),
                    None)
                s_col = next(
                    (col for col in rows[0].keys() if
                     col.lower() in ['golden answer sections', 'section_ids', 'sections']),
                    None)

                if not q_col or not r_col:
                    st.error(f"❌ Could not find required columns. Available: {', '.join(rows[0].keys())}")
                    st.stop()

                # Preview data
                st.markdown("#### 👀 Data Preview")
                preview_cols = [q_col, r_col]
                if s_col:
                    preview_cols.append(s_col)
                preview_df = pd.DataFrame(rows[:5])[preview_cols]
                st.dataframe(preview_df, use_container_width=True)
                st.info(f"📊 Found **{len(rows)} questions** to evaluate")

                # Run evaluation button
                if st.button("▶️ Run Evaluation", type="primary", use_container_width=True):
                    evaluator = st.session_state.judge_evaluator
                    evaluator.judge_llm = OllamaLLM(model=judge_model, temperature=0.0)

                    results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # single event loop for all rows
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    # Evaluate both systems
                    for i, row in enumerate(rows):
                        status_text.text(f"Processing {i + 1}/{len(rows)}: {row[q_col][:50]}...")
                        progress_bar.progress((i + 1) / len(rows))

                        # Evaluate Standard RAG
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        section_ids = row.get(s_col, None) if s_col else None

                        std_task = evaluator.evaluate_single_query(
                                row[q_col],
                                row[r_col],
                                "Standard RAG",
                                answer_section_ids=section_ids
                            )

                        graph_task = evaluator.evaluate_single_query(
                                row[q_col],
                                row[r_col],
                                "GraphRAG",
                                answer_section_ids=section_ids
                            )

                        std_result, graph_result = loop.run_until_complete(
                            asyncio.gather(std_task, graph_task)
                        )

                        std_result['system'] = 'Standard RAG'
                        std_result['question'] = row[q_col]
                        std_result['golden_answer'] = row[r_col]
                        results.append(std_result)

                        graph_result['system'] = 'GraphRAG'
                        graph_result['question'] = row[q_col]
                        graph_result['golden_answer'] = row[r_col]
                        results.append(graph_result)

                    loop.close()

                    status_text.text("✅ Evaluation complete!")
                    progress_bar.empty()
                    status_text.empty()

                    # ================================================================
                    # APPROACH 1: COMPREHENSIVE MULTI-METRIC DASHBOARD
                    # ================================================================

                    st.markdown("---")
                    st.markdown("## 📊 Evaluation Results Dashboard")

                    # Convert to DataFrame
                    df = pd.DataFrame(results)

                    # Separate by system
                    std_df = df[df['system'] == 'Standard RAG']
                    graph_df = df[df['system'] == 'GraphRAG']

                    # ============================================================
                    # SECTION 1: SUMMARY METRICS PANEL (CORRECTED)
                    # ============================================================
                    st.markdown("### 🎯 Overall Performance Summary")

                    # Convert to DataFrame
                    df = pd.DataFrame(results)

                    # Separate by system
                    std_df = df[df['system'] == 'Standard RAG']
                    graph_df = df[df['system'] == 'GraphRAG']

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        std_avg = std_df['total_score'].mean()
                        st.metric(
                            "Standard RAG",
                            f"{std_avg:.1f}",
                            help="Average total score (LLM Score 50% + Section Match 50%)"
                        )

                    with col2:
                        graph_avg = graph_df['total_score'].mean()
                        delta = graph_avg - std_avg
                        st.metric(
                            "GraphRAG",
                            f"{graph_avg:.1f}",
                            delta=f"{delta:+.1f}",
                            delta_color="normal",
                            help="Average total score (LLM Score 50% + Section Match 50%)"
                        )

                    with col3:
                        std_success = (std_df['total_score'] >= 70).sum() / len(std_df) * 100
                        graph_success = (graph_df['total_score'] >= 70).sum() / len(graph_df) * 100
                        st.metric(
                            "Success Rate (≥70)",
                            f"{graph_success:.1f}%",
                            delta=f"{graph_success - std_success:+.1f}%",
                            help="Percentage of queries scoring 70 or above"
                        )

                    with col4:
                        std_time = std_df['response_time'].median()
                        graph_time = graph_df['response_time'].median()
                        st.metric(
                            "Median Response Time",
                            f"{graph_time:.2f}s",
                            delta=f"{graph_time - std_time:+.2f}s",
                            delta_color="inverse",
                            help="Median query response time"
                        )

                    # Detailed score breakdown
                    st.markdown("#### 📈 Score Component Breakdown")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown("**LLM Judge Score** (50% weight)")
                        subcol1, subcol2 = st.columns(2)
                        subcol1.metric("Standard RAG", f"{std_df['llm_score'].mean():.1f}")
                        subcol2.metric("GraphRAG", f"{graph_df['llm_score'].mean():.1f}")

                    with col2:
                        st.markdown("**Section Match Score** (50% weight)")
                        subcol1, subcol2 = st.columns(2)
                        subcol1.metric("Standard RAG", f"{std_df['section_score'].mean():.1f}")
                        subcol2.metric("GraphRAG", f"{graph_df['section_score'].mean():.1f}")

                    with col3:
                        st.markdown("**Sections Retrieved (Avg)**")
                        subcol1, subcol2 = st.columns(2)
                        # FIXED: Use retrieved_sections instead of num_sections
                        if 'retrieved_sections' in std_df.columns:
                            # Count sections in the list
                            std_sections_count = std_df['retrieved_sections'].apply(
                                lambda x: len(x) if isinstance(x, list) else 0).mean()
                            graph_sections_count = graph_df['retrieved_sections'].apply(
                                lambda x: len(x) if isinstance(x, list) else 0).mean()
                            subcol1.metric("Standard RAG", f"{std_sections_count:.1f}")
                            subcol2.metric("GraphRAG", f"{graph_sections_count:.1f}")
                        else:
                            subcol1.metric("Standard RAG", "N/A")
                            subcol2.metric("GraphRAG", "N/A")

                    st.markdown("---")

                    # ============================================================
                    # SECTION 2: DETAILED RESULTS TABLE
                    # ============================================================
                    st.markdown("### 📋 Detailed Query-by-Query Results")

                    # Create comparison dataframe
                    comparison_data = []
                    for question in df['question'].unique():
                        std_row = std_df[std_df['question'] == question].iloc[0]
                        graph_row = graph_df[graph_df['question'] == question].iloc[0]

                        # FIXED: Handle retrieved_sections safely
                        std_sections = len(std_row.get('retrieved_sections', [])) if isinstance(
                            std_row.get('retrieved_sections'), list) else 0
                        graph_sections = len(graph_row.get('retrieved_sections', [])) if isinstance(
                            graph_row.get('retrieved_sections'), list) else 0

                        comparison_data.append({
                            'Question': question[:60] + '...' if len(question) > 60 else question,
                            'Std RAG Score': f"{std_row['total_score']:.1f}",
                            'GraphRAG Score': f"{graph_row['total_score']:.1f}",
                            'Winner': '🏆 GraphRAG' if graph_row['total_score'] > std_row['total_score']
                            else '🏆 Std RAG' if std_row['total_score'] > graph_row['total_score']
                            else '🤝 Tie',
                            'Δ Score': f"{graph_row['total_score'] - std_row['total_score']:+.1f}",
                            'Std Sections': std_sections,
                            'Graph Sections': graph_sections,
                            'Std Time': f"{std_row['response_time']:.2f}s",
                            'Graph Time': f"{graph_row['response_time']:.2f}s"
                        })

                    comparison_df = pd.DataFrame(comparison_data)

                    # Style the dataframe
                    def highlight_winner(row):
                        if '🏆 GraphRAG' in row['Winner']:
                            return ['background-color: #d4edda'] * len(row)
                        elif '🏆 Std RAG' in row['Winner']:
                            return ['background-color: #fff3cd'] * len(row)
                        else:
                            return [''] * len(row)

                    styled_df = comparison_df.style.apply(highlight_winner, axis=1)
                    st.dataframe(styled_df, use_container_width=True, height=400)

                    st.markdown("---")

                    # ============================================================
                    # SECTION 3: STATISTICAL ANALYSIS
                    # ============================================================
                    st.markdown("### 📊 Statistical Analysis")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("#### Score Distribution")

                        # Create histogram data
                        score_bins = [0, 50, 70, 90, 100]
                        score_labels = ['0-50 (Poor)', '50-70 (Fair)', '70-90 (Good)', '90-100 (Excellent)']

                        std_binned = pd.cut(std_df['total_score'], bins=score_bins, labels=score_labels)
                        graph_binned = pd.cut(graph_df['total_score'], bins=score_bins, labels=score_labels)

                        dist_df = pd.DataFrame({
                            'Score Range': score_labels * 2,
                            'Count': list(std_binned.value_counts().sort_index()) + list(
                                graph_binned.value_counts().sort_index()),
                            'System': ['Standard RAG'] * 4 + ['GraphRAG'] * 4
                        })

                        st.dataframe(
                            dist_df.pivot(index='Score Range', columns='System', values='Count').fillna(0).astype(int),
                            use_container_width=True)

                    with col2:
                        st.markdown("#### Win/Loss/Tie Breakdown")

                        wins = (graph_df['total_score'].values > std_df['total_score'].values).sum()
                        losses = (graph_df['total_score'].values < std_df['total_score'].values).sum()
                        ties = (graph_df['total_score'].values == std_df['total_score'].values).sum()
                        total = len(std_df)

                        wlt_df = pd.DataFrame({
                            'Outcome': ['GraphRAG Wins', 'Standard RAG Wins', 'Ties'],
                            'Count': [wins, losses, ties],
                            'Percentage': [f"{wins / total * 100:.1f}%", f"{losses / total * 100:.1f}%",
                                           f"{ties / total * 100:.1f}%"]
                        })

                        st.dataframe(wlt_df, use_container_width=True, hide_index=True)

                        # Visual representation
                        st.markdown(f"""
                        - **GraphRAG Superior:** {wins} queries ({wins / total * 100:.1f}%)
                        - **Standard RAG Superior:** {losses} queries ({losses / total * 100:.1f}%)
                        - **Comparable:** {ties} queries ({ties / total * 100:.1f}%)
                        """)

                    # Performance by score delta
                    st.markdown("#### Performance Gap Analysis")
                    col1, col2, col3 = st.columns(3)

                    score_diff = graph_df['total_score'].values - std_df['total_score'].values

                    with col1:
                        large_wins = (score_diff > 20).sum()
                        st.metric(
                            "Large GraphRAG Wins (Δ>20)",
                            large_wins,
                            help="Queries where GraphRAG scored 20+ points higher"
                        )

                    with col2:
                        large_losses = (score_diff < -20).sum()
                        st.metric(
                            "Large Standard RAG Wins (Δ>20)",
                            large_losses,
                            help="Queries where Standard RAG scored 20+ points higher"
                        )

                    with col3:
                        comparable = ((score_diff >= -10) & (score_diff <= 10)).sum()
                        st.metric(
                            "Comparable Performance (|Δ|≤10)",
                            comparable,
                            help="Queries with similar scores (within 10 points)"
                        )

                    st.markdown("---")

                    # ============================================================
                    # DETAILED ANSWER COMPARISON
                    # ============================================================
                    st.markdown("### 🔍 Detailed Answer Quality Analysis")
                    st.markdown("Examine actual answers, scores, and LLM judge reasoning for each query.")

                    # Create side-by-side comparison for each question
                    for idx, question in enumerate(df['question'].unique(), 1):
                        std_row = std_df[std_df['question'] == question].iloc[0]
                        graph_row = graph_df[graph_df['question'] == question].iloc[0]

                        # Determine winner
                        if graph_row['total_score'] > std_row['total_score']:
                            winner_badge = "🏆 **GraphRAG Wins**"
                            winner_color = "green"
                        elif std_row['total_score'] > graph_row['total_score']:
                            winner_badge = "🏆 **Standard RAG Wins**"
                            winner_color = "orange"
                        else:
                            winner_badge = "🤝 **Tie**"
                            winner_color = "blue"

                        # Expandable section for each query
                        with st.expander(f"**Query {idx}:** {question[:80]}{'...' if len(question) > 80 else ''}",
                                         expanded=(idx == 1)):

                            # Header with winner badge
                            st.markdown(f":{winner_color}[{winner_badge}]")
                            st.markdown("---")

                            # ===== QUESTION AND REFERENCE =====
                            st.markdown("#### 📝 Question & Reference Answer")
                            st.info(f"**Question:** {question}")
                            st.success(f"**Reference (Golden) Answer:** {std_row['golden_answer']}")

                            st.markdown("---")

                            # ===== SIDE-BY-SIDE COMPARISON =====
                            col1, col2 = st.columns(2)

                            with col1:
                                st.markdown("### 🔵 Standard RAG")

                                # Score Summary
                                st.markdown("**📊 Scores**")
                                score_col1, score_col2, score_col3 = st.columns(3)
                                score_col1.metric("Total", f"{std_row['total_score']:.1f}")
                                score_col2.metric("LLM", f"{std_row['llm_score']:.1f}")
                                score_col3.metric("Section", f"{std_row['section_score']:.1f}")

                                # Predicted Answer
                                st.markdown("**💬 Generated Answer**")
                                st.text_area(
                                    "Standard RAG Response",
                                    std_row['predicted_answer'],
                                    height=150,
                                    key=f"std_answer_{idx}",
                                    label_visibility="collapsed"
                                )

                                # Retrieved Sections
                                st.markdown("**📚 Retrieved Sections**")
                                retrieved_sections = std_row.get('retrieved_sections', [])
                                if retrieved_sections and isinstance(retrieved_sections, list):
                                    st.caption(
                                        f"🔖 {len(retrieved_sections)} sections: {', '.join(map(str, retrieved_sections))}")
                                else:
                                    st.caption("🔖 No sections retrieved")

                                # LLM Judge Explanation
                                st.markdown("**🤖 LLM Judge Reasoning**")
                                st.text_area(
                                    "Judge Explanation (Std)",
                                    std_row.get('llm_score_explanation', 'No explanation provided'),
                                    height=100,
                                    key=f"std_explanation_{idx}",
                                    label_visibility="collapsed"
                                )

                                # Response Time
                                st.caption(f"⏱️ Response Time: {std_row['response_time']:.2f}s")

                            with col2:
                                st.markdown("### 🟢 GraphRAG")

                                # Score Summary
                                st.markdown("**📊 Scores**")
                                score_col1, score_col2, score_col3 = st.columns(3)
                                score_col1.metric(
                                    "Total",
                                    f"{graph_row['total_score']:.1f}",
                                    delta=f"{graph_row['total_score'] - std_row['total_score']:+.1f}"
                                )
                                score_col2.metric(
                                    "LLM",
                                    f"{graph_row['llm_score']:.1f}",
                                    delta=f"{graph_row['llm_score'] - std_row['llm_score']:+.1f}"
                                )
                                score_col3.metric(
                                    "Section",
                                    f"{graph_row['section_score']:.1f}",
                                    delta=f"{graph_row['section_score'] - std_row['section_score']:+.1f}"
                                )

                                # Predicted Answer
                                st.markdown("**💬 Generated Answer**")
                                st.text_area(
                                    "GraphRAG Response",
                                    graph_row['predicted_answer'],
                                    height=150,
                                    key=f"graph_answer_{idx}",
                                    label_visibility="collapsed"
                                )

                                # Retrieved Sections
                                st.markdown("**📚 Retrieved Sections**")
                                retrieved_sections = graph_row.get('retrieved_sections', [])
                                if retrieved_sections and isinstance(retrieved_sections, list):
                                    st.caption(
                                        f"🔖 {len(retrieved_sections)} sections: {', '.join(map(str, retrieved_sections))}")
                                else:
                                    st.caption("🔖 No sections retrieved")

                                # LLM Judge Explanation
                                st.markdown("**🤖 LLM Judge Reasoning**")
                                st.text_area(
                                    "Judge Explanation (Graph)",
                                    graph_row.get('llm_score_explanation', 'No explanation provided'),
                                    height=100,
                                    key=f"graph_explanation_{idx}",
                                    label_visibility="collapsed"
                                )

                                # Response Time
                                st.caption(f"⏱️ Response Time: {graph_row['response_time']:.2f}s")

                            # ===== COMPARATIVE ANALYSIS =====
                            st.markdown("---")
                            st.markdown("#### 📈 Comparative Analysis")

                            analysis_col1, analysis_col2, analysis_col3 = st.columns(3)

                            with analysis_col1:
                                score_diff = graph_row['total_score'] - std_row['total_score']
                                if score_diff > 10:
                                    st.success(f"✅ GraphRAG significantly better (+{score_diff:.1f})")
                                elif score_diff < -10:
                                    st.warning(f"⚠️ Standard RAG significantly better ({score_diff:.1f})")
                                else:
                                    st.info(f"➡️ Comparable performance ({score_diff:+.1f})")

                            with analysis_col2:
                                std_sections = len(std_row.get('retrieved_sections', []))
                                graph_sections = len(graph_row.get('retrieved_sections', []))
                                section_diff = graph_sections - std_sections
                                st.metric(
                                    "Section Retrieval Difference",
                                    graph_sections,
                                    delta=f"{section_diff:+d} vs Std RAG"
                                )

                            with analysis_col3:
                                time_diff = graph_row['response_time'] - std_row['response_time']
                                time_diff_pct = (time_diff / std_row['response_time']) * 100
                                st.metric(
                                    "Response Time Difference",
                                    f"{graph_row['response_time']:.2f}s",
                                    delta=f"{time_diff:+.2f}s ({time_diff_pct:+.0f}%)",
                                    delta_color="inverse"
                                )

                    st.markdown("---")

                    # ============================================================
                    # SECTION 4: ADDITIONAL INSIGHTS (CORRECTED)
                    # ============================================================
                    st.markdown("### 💡 Key Insights")

                    # Calculate insights - FIXED
                    if 'retrieved_sections' in std_df.columns:
                        avg_std_sections = std_df['retrieved_sections'].apply(
                            lambda x: len(x) if isinstance(x, list) else 0).mean()
                        avg_graph_sections = graph_df['retrieved_sections'].apply(
                            lambda x: len(x) if isinstance(x, list) else 0).mean()
                    else:
                        avg_std_sections = 0
                        avg_graph_sections = 0

                    section_match_improvement = graph_df['section_score'].mean() - std_df['section_score'].mean()
                    llm_score_improvement = graph_df['llm_score'].mean() - std_df['llm_score'].mean()

                    col1, col2 = st.columns(2)

                    with col1:
                        st.info(f"""
                        **📚 Retrieval Analysis:**
                        - Standard RAG retrieved {avg_std_sections:.1f} sections on average
                        - GraphRAG retrieved {avg_graph_sections:.1f} sections on average
                        - Section match score improved by **{section_match_improvement:+.1f} points**
                        """)

                    with col2:
                        st.success(f"""
                        **🎯 Answer Quality:**
                        - LLM judge score improved by **{llm_score_improvement:+.1f} points**
                        - GraphRAG won **{wins}/{total} queries** ({wins / total * 100:.1f}%)
                        - Overall score improvement: **{delta:+.1f} points**
                        """)

                    # Response time analysis
                    if graph_time > std_time:
                        time_penalty = ((graph_time - std_time) / std_time) * 100
                        st.warning(f"""
                        **⏱️ Performance Trade-off:**
                        GraphRAG is {time_penalty:.1f}% slower ({graph_time - std_time:.2f}s additional latency) 
                        but provides {delta:+.1f} points higher accuracy on average.
                        """)

                    st.markdown("---")

                    # ============================================================
                    # SECTION 5: EXPORT OPTIONS
                    # ============================================================
                    st.markdown("### 💾 Export Results")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        # Export full results
                        csv_data = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Full Results (CSV)",
                            data=csv_data,
                            file_name=f"evaluation_results_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                    with col2:
                        # Export comparison table
                        comparison_csv = comparison_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Comparison (CSV)",
                            data=comparison_csv,
                            file_name=f"comparison_table_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                    with col3:
                        # Export summary statistics
                        summary_data = {
                            'Metric': ['Avg Total Score', 'Avg LLM Score', 'Avg Section Score',
                                       'Success Rate (≥70)', 'Median Response Time'],
                            'Standard RAG': [std_avg, std_df['llm_score'].mean(), std_df['section_score'].mean(),
                                             std_success, std_time],
                            'GraphRAG': [graph_avg, graph_df['llm_score'].mean(), graph_df['section_score'].mean(),
                                         graph_success, graph_time]
                        }
                        summary_df = pd.DataFrame(summary_data)
                        summary_csv = summary_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Summary (CSV)",
                            data=summary_csv,
                            file_name=f"summary_stats_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    st.stop()




            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.exception(e)


if __name__ == "__main__":
    main()
