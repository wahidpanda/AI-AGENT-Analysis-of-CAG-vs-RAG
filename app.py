import streamlit as st
import pandas as pd
import plotly.express as px
import time
import json
import asyncio
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Set up paths
RESULTS_DIR = Path(__file__).parent / "Results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize session state variables
if "history" not in st.session_state:
    st.session_state.history = []
if "cache_hits" not in st.session_state:
    st.session_state.cache_hits = 0
if "cache_misses" not in st.session_state:
    st.session_state.cache_misses = 0
if "response_times" not in st.session_state:
    st.session_state.response_times = []
if "query_timestamps" not in st.session_state:
    st.session_state.query_timestamps = []
if "cache_content" not in st.session_state:
    st.session_state.cache_content = {}

# Page title
st.set_page_config(
    page_title="CAG vs RAG Framework Comparison",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# CSS for Styling
st.markdown(
    """
    <style>
        body { font-family: 'Arial', sans-serif; }
        .stTextInput, .stButton { border-radius: 8px; }
        .stProgress > div > div { border-radius: 20px; }
        .custom-link { color: #1f77b4; text-decoration: none; font-weight: bold; transition: color 0.3s ease-in-out; }
        .custom-link:hover { color: #ff4b4b; }
        .fixed-graph-container { max-height: 300px !important; overflow-y: auto; }
        .response-container { border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
        .response-container h4 { margin-top: 0; }
    </style>
    """,
    unsafe_allow_html=True
)

# Page Title and Description
st.title("💡 Cache Augmented Generation (CAG) vs Retrieval Augmented Generation (RAG)")
st.write("**Compare the performance of CAG and RAG frameworks with enhanced caching.**")

# Layout Columns: Configurator | Chat | Statistics
col1, col2, col3 = st.columns([1.2, 2, 1.2])

# 🛠️ **Configurator Section (Left Panel)**
with col1:
    st.header("⚙️ Configurator")
    cache_size = st.slider("🗄️ Cache Size", min_value=50, max_value=500, value=100)
    similarity_threshold = st.slider("📈 Similarity Threshold", min_value=0.5, max_value=1.0, value=0.8)
    clear_cache = st.button("🧹 Clear Cache")

    if clear_cache:
        st.session_state.cache_hits = 0
        st.session_state.cache_misses = 0
        st.session_state.response_times = []
        st.session_state.query_timestamps = []
        st.session_state.history = []
        st.session_state.cache_content = {}
        st.success("✅ Cache cleared successfully!")

    # 📦 **Cache Content Section**
    with st.expander("📦 **View Cache Content**"):
        if st.session_state.cache_content:
            for key, value in st.session_state.cache_content.items():
                st.write(f"**Query:** {key}")
                st.write(f"**Response:** {value['response']}")
                st.write(f"**Timestamp:** {datetime.fromtimestamp(value['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
                st.write("---")
        else:
            st.write("🗑️ Cache is currently empty.")

# 💬 **Chat Interaction Section (Middle Panel)**
with col2:
    st.header("💬 Chat with CAG")
    query = st.text_input("💡 Enter your query:")
    if query:
        start_time = time.time()

        # Step 1: Check Cache
        st.info("⏳ Checking Cache...")
        cached_response = None  # Replace with actual cache logic
        if cached_response:
            # Step 2: If Cache Hit, Return
            st.success("✅ Cache Hit! Returning cached response.")
            response = cached_response
            st.session_state.cache_hits += 1
        else:
            # Step 3: If Cache Miss, Query LLM
            st.warning("❌ Cache Miss. Fetching from LLM...")
            response = "Simulated response."  # Replace with actual LLM call
            st.session_state.cache_misses += 1

        # Response Time and Save Data
        response_time = time.time() - start_time
        st.session_state.response_times.append(response_time)
        st.session_state.query_timestamps.append(datetime.now().strftime('%H:%M:%S'))
        st.session_state.history.append({"query": query, "response": response, "time": response_time})

        # 🎯 Chat Response
        st.success(f"**🗨️ {response}**")
        st.info(f"⏱️ **Response Time:** {response_time:.2f} seconds")

    # 📜 **Query History Section**
    with st.expander("🕰️ **Query History**"):
        for entry in st.session_state.history[-10:]:
            st.write(f"**Query:** {entry['query']}")
            st.write(f"**Response:** {entry['response']}")
            st.write(f"⏱️ **Time Taken:** {entry['time']:.2f} seconds")
            st.write("---")

# 📊 **Cache Statistics Section (Right Panel)**
with col3:
    st.header("📊 Cache Statistics")

    # Real-Time Metrics
    col1_stat, col2_stat, col3_stat = st.columns(3)
    col1_stat.metric("✅ Hits", st.session_state.cache_hits)
    col2_stat.metric("❌ Misses", st.session_state.cache_misses)
    col3_stat.metric("📦 Cache Size", len(st.session_state.cache_content))

    # Cache Hit/Miss Ratio
    total_queries = st.session_state.cache_hits + st.session_state.cache_misses
    hit_ratio = (st.session_state.cache_hits / total_queries) * 100 if total_queries > 0 else 0
    miss_ratio = (st.session_state.cache_misses / total_queries) * 100 if total_queries > 0 else 0

    st.progress(hit_ratio / 100, text=f"✅ Cache Hit Ratio: {hit_ratio:.2f}%")
    st.progress(miss_ratio / 100, text=f"❌ Cache Miss Ratio: {miss_ratio:.2f}%")

    # 📈 **Response Time Graph**
    if st.session_state.response_times:
        st.markdown('<div class="fixed-graph-container">', unsafe_allow_html=True)
        fig = px.area(
            x=st.session_state.query_timestamps,
            y=st.session_state.response_times,
            title="📈 Response Time Trend",
            labels={"x": "Timestamp", "y": "Response Time (s)"},
            line_shape="spline"  # Smoothed line
        )
        fig.update_traces(fill="tozeroy", line=dict(color="#1f77b4"))  # Fill area under the curve
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="black"),
            xaxis=dict(showgrid=True, gridcolor="#eee"),
            yaxis=dict(showgrid=True, gridcolor="#eee"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ✅ **Footer**
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center;">
        <p><strong>🚀 Built for the demonstration of Cache Augmented Generation.</strong></p>
    </div>
    """, 
    unsafe_allow_html=True
)

# CAG vs RAG Comparison Tab
st.header("CAG vs RAG Comparison")

# Load existing results (if any)
def load_results():
    """Load existing results from JSON files."""
    results = []
    for result_file in RESULTS_DIR.glob("comparison_results_*.json"):
        with open(result_file, "r") as f:
            results.append(json.load(f))
    return results

# Save new results
def save_results(query, cag_response, rag_response, cag_time, rag_time, cag_metrics, rag_metrics):
    """Save comparison results to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"comparison_results_{timestamp}.json"
    
    result = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "cag_response": cag_response,
        "rag_response": rag_response,
        "cag_time": cag_time,
        "rag_time": rag_time,
        "time_difference": rag_time - cag_time,
        "cag_metrics": cag_metrics,  # Ensure this is a dictionary
        "rag_metrics": rag_metrics   # Ensure this is a dictionary
    }
    
    with open(results_file, "w") as f:
        json.dump(result, f, indent=2)
    
    return result

# Display results
def display_results(results):
    """Display comparison results in a table and charts."""
    if not results:
        st.warning("No results found. Run a comparison first.")
        return
    
    # Convert results to a DataFrame
    df = pd.DataFrame(results)
    
    # Show raw data
    st.subheader("Raw Data")
    st.write(df)
    
    # Show performance metrics
    st.subheader("Performance Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Average CAG Response Time", f"{df['cag_time'].mean():.2f} seconds")
    
    with col2:
        st.metric("Average RAG Response Time", f"{df['rag_time'].mean():.2f} seconds")
    
    with col3:
        st.metric("Average Time Difference (RAG - CAG)", f"{df['time_difference'].mean():.2f} seconds")
    
    # Plot response times
    st.subheader("Response Time Comparison")
    fig, ax = plt.subplots()
    ax.plot(df["timestamp"], df["cag_time"], label="CAG", marker="o")
    ax.plot(df["timestamp"], df["rag_time"], label="RAG", marker="o")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Response Time (seconds)")
    ax.legend()
    st.pyplot(fig)
    
    # Plot cache hits (if available)
    if "cag_metrics" in df.columns:
        # Check if cag_metrics is a dictionary and contains "cache_hits"
        if isinstance(df["cag_metrics"].iloc[0], dict) and "cache_hits" in df["cag_metrics"].iloc[0]:
            cache_hits = [metrics.get("cache_hits", 0) for metrics in df["cag_metrics"]]
            fig, ax = plt.subplots()
            ax.bar(df["timestamp"], cache_hits, label="Cache Hits", color="orange")
            ax.set_xlabel("Timestamp")
            ax.set_ylabel("Cache Hits")
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("Cache hits data is not available or is in an unexpected format.")

# Simulate CAG and RAG responses (replace with actual API calls)
async def simulate_cag_response(query):
    """Simulate a CAG response."""
    await asyncio.sleep(1)  # Simulate async delay
    return f"CAG Response to: {query}", {"cache_hits": 1, "response_time": 0.5, "memory_usage": 3}

async def simulate_rag_response(query):
    """Simulate a RAG response."""
    await asyncio.sleep(1)  # Simulate async delay
    return f"RAG Response to: {query}", {"response_time": 0.7, "num_retrieved": 1, "retriever_type": "hybrid"}

# User input for comparison
query = st.text_input("Enter your query for comparison:", "What are the main advantages of the CAG framework?")

if st.button("Run Comparison"):
    st.subheader("Running Comparison...")
    
    # Run async functions using asyncio.run
    async def run_comparison():
        cag_response, cag_metrics = await simulate_cag_response(query)
        rag_response, rag_metrics = await simulate_rag_response(query)
        return cag_response, cag_metrics, rag_response, rag_metrics
    
    cag_response, cag_metrics, rag_response, rag_metrics = asyncio.run(run_comparison())
    
    # Calculate response times (simulated)
    cag_time = cag_metrics["response_time"]
    rag_time = rag_metrics["response_time"]
    
    # Save results
    result = save_results(query, cag_response, rag_response, cag_time, rag_time, cag_metrics, rag_metrics)
    
    # Display results
    st.subheader("Comparison Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="response-container">', unsafe_allow_html=True)
        st.write("**CAG Response:**")
        st.write(cag_response)
        st.write(f"**Time taken:** {cag_time:.2f} seconds")
        st.write("**Metrics:**", cag_metrics)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="response-container">', unsafe_allow_html=True)
        st.write("**RAG Response:**")
        st.write(rag_response)
        st.write(f"**Time taken:** {rag_time:.2f} seconds")
        st.write("**Metrics:**", rag_metrics)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.success("Comparison completed!")

# Load and display historical results
st.subheader("Historical Results")
results = load_results()
display_results(results)