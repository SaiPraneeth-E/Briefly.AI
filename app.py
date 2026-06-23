import os
import io
import time
import streamlit as st
import pandas as pd

# Plotly is imported lazily when charts are rendered (saves ~1.5s on first load)
px = None
go = None

def _ensure_plotly():
    """Lazy-load plotly on first use to speed up initial page render."""
    global px, go
    if px is None:
        import plotly.express as _px
        import plotly.graph_objects as _go
        px = _px
        go = _go

from utils import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_txt,
    clean_text,
    get_document_stats,
    get_pdf_download_bytes,
    get_docx_download_bytes
)
from summarizer import DocumentSummarizerPipeline, clean_gpu_memory, get_device_str

# Set page configuration with a custom title and wide layout
st.set_page_config(
    page_title="AI Document Summarizer & Analytics",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling using CSS injection
st.markdown("""
<style>
    /* Google Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');
    
    /* Main body typography */
    html, body, [class*="css"], .stMarkdown, p, li, span, label {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #94A3B8 !important;
    }
    
    h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, strong {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        color: #F8FAFC !important;
    }
    
    /* Premium background decoration */
    .stApp {
        background-color: #0B0F19 !important;
    }
    
    /* Sidebar styling override */
    section[data-testid="stSidebar"] {
        background-color: #060911 !important;
        border-right: 1px solid #1E293B;
    }
    
    /* Text inputs, select box, sliders dark theme integration */
    div[data-baseweb="select"] > div, input, textarea {
        background-color: #131A2C !important;
        color: #F8FAFC !important;
        border-color: #1E293B !important;
    }
    
    /* Elegant App Banner */
    .banner-container {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #C084FC 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white !important;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(124, 58, 237, 0.3);
    }
    
    .banner-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.05em;
        background: linear-gradient(to right, #FFFFFF, #E0E7FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .banner-subtitle {
        font-size: 1.1rem;
        font-weight: 300;
        color: #E0E7FF !important;
        max-width: 600px;
        margin: 0 auto;
    }
    
    /* Custom Styling for KPI metrics */
    .kpi-card {
        background-color: #131A2C !important;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2), 0 2px 4px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #1E293B !important;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .kpi-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -2px rgba(0, 0, 0, 0.2);
        border-color: #4F46E5 !important;
    }
    
    .kpi-val {
        font-size: 2.2rem;
        font-weight: 700;
        color: #818CF8 !important; /* light indigo */
        margin-bottom: 0.2rem;
        line-height: 1;
    }
    
    .kpi-lbl {
        font-size: 0.85rem;
        color: #64748B !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* File uploader styling refinement */
    .uploadedFile {
        border-radius: 12px;
        border: 2px dashed #1E293B !important;
        background-color: #131A2C !important;
    }
    
    /* Rounded buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: white !important;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.4);
        transition: all 0.3s ease;
    }
    
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #4338CA 0%, #6D28D9 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(99, 102, 241, 0.5);
    }
    
    /* Section dividers */
    .divider {
        height: 2px;
        background: linear-gradient(to right, #0F172A, #1E293B, #0F172A);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Export helper functions are imported from utils.py

# Initialize State
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'stats' not in st.session_state:
    st.session_state.stats = {}
if 'keywords' not in st.session_state:
    st.session_state.keywords = []
if 'topics' not in st.session_state:
    st.session_state.topics = {}
if 'sentiment' not in st.session_state:
    st.session_state.sentiment = {}
if 'is_processed' not in st.session_state:
    st.session_state.is_processed = False
if 'file_name' not in st.session_state:
    st.session_state.file_name = ""
if 'pipeline_cache' not in st.session_state:
    st.session_state.pipeline_cache = {}
if 'pdf_bytes' not in st.session_state:
    st.session_state.pdf_bytes = b""
if 'docx_bytes' not in st.session_state:
    st.session_state.docx_bytes = b""
if 'last_exported_summary' not in st.session_state:
    st.session_state.last_exported_summary = ""
if 'processed_docs' not in st.session_state:
    st.session_state.processed_docs = {}
if 'batch_zip' not in st.session_state:
    st.session_state.batch_zip = b""
if 'selected_doc_name' not in st.session_state:
    st.session_state.selected_doc_name = ""

# Page Banner
st.markdown("""
<div class="banner-container">
    <div class="banner-title">AI Document Summarizer & Insights</div>
    <div class="banner-subtitle">Transform long PDFs, DOCX, and text articles into high-quality summarized reports, topics, keywords, and sentiment models instantly.</div>
</div>
""", unsafe_allow_html=True)

# Safe helper to check st.secrets or env variables
def get_secret(key, default=""):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)

# Sidebar layout
st.sidebar.markdown("### 🔑 API Authentication (Optional)")

gemini_default = get_secret("GEMINI_API_KEY")
gemini_key = st.sidebar.text_input(
    "Google Gemini API Key",
    value=gemini_default,
    type="password",
    help="Provide a free Google Gemini API Key to run summaries remotely on Gemini 1.5 Flash. This executes the entire NLP pipeline in under 1.5 seconds, requires 0MB local RAM, and supports massive documents."
)

hf_default = get_secret("HF_API_TOKEN")
hf_token = st.sidebar.text_input(
    "Hugging Face User Access Token",
    value=hf_default,
    type="password",
    help="Provide a Hugging Face Access Token (starts with hf_) to run models remotely via HF Serverless Inference API (under 3s)."
)

if gemini_key:
    st.sidebar.success("🚀 Gemini 1.5 Flash API Active")
elif hf_token:
    st.sidebar.success("🌐 HF Cloud API Active")
else:
    st.sidebar.info("💻 Local PyTorch Mode Active")
    st.sidebar.warning(
        "⚠️ **Performance Notice:** Running in Local CPU mode can be extremely slow and might crash on free hosting platforms (like Streamlit Cloud) due to RAM limits. Provide a Gemini API Key or Hugging Face Token for fast and stable cloud execution."
    )

st.sidebar.markdown("### 🛠️ Configuration Panel")

# Model configuration
model_option = st.sidebar.selectbox(
    "Primary Summarizer Model",
    options=[
        "sshleifer/distilbart-cnn-12-6",
        "facebook/bart-large-cnn",
        "google/pegasus-cnn_dailymail",
        "google/flan-t5-large"
    ],
    index=0,
    help="Select the pretrained model. DistilBART is recommended for speed and low RAM consumption."
)

# Performance selection
perf_mode = st.sidebar.radio(
    "Hardware & Resource Mode",
    options=["Low Resource (Fast & Light)", "High Performance (Accurate)"],
    index=0,
    help="Low Resource mode uses lighter NLP components suitable for free Cloud hosting."
)

# Select zero-shot model depending on performance mode
classifier_model = "valhalla/distilbart-mnli-12-6" if perf_mode.startswith("Low") else "facebook/bart-large-mnli"

if model_option == "google/pegasus-cnn_dailymail" and not hf_token:
    st.sidebar.warning("⚠️ Pegasus is a large model (2.3GB) and is extremely slow on local CPU. Consider using DistilBART or providing a Hugging Face Access Token above.")

# Summary configuration
summary_mode = st.sidebar.selectbox(
    "Summary Structuring Mode",
    options=[
        "Executive Summary",
        "Bullet Point Summary",
        "Detailed Summary",
        "Meeting Notes Summary",
        "Research Paper Summary",
        "Key Insights Summary"
    ],
    index=0
)

summary_length = st.sidebar.select_slider(
    "Summary Target Length",
    options=["short", "medium", "long"],
    value="medium"
)

# Advanced parameters
with st.sidebar.expander("⚙️ Advanced Generation Settings"):
    temp = st.slider("Temperature (Flan-T5 sampling)", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
    beams = st.slider("Beam Search Count", min_value=1, max_value=5, value=4, step=1)
    penalty = st.slider("Length Penalty", min_value=1.0, max_value=3.0, value=2.0, step=0.5)

# Fast analysis options
st.sidebar.markdown("### ⚡ Performance & Analysis Settings")
fast_mode = st.sidebar.checkbox(
    "Fast Extraction Mode",
    value=True,
    help="Run topic detection, sentiment analysis, and keyword extraction on the summary rather than the full document (10x faster with comparable accuracy)."
)
enable_topic = st.sidebar.checkbox("Enable Topic Detection", value=True)
enable_sentiment = st.sidebar.checkbox("Enable Sentiment Analysis", value=True)
enable_keywords = st.sidebar.checkbox("Enable Keyword Extraction", value=True)

# Topics configuration
st.sidebar.markdown("### 🏷️ Topic Classification Settings")
custom_topics_str = st.sidebar.text_input(
    "Candidate Topics (comma-separated)",
    value="Technology, Finance, Healthcare, Education, Research, Business, Legal, Government"
)
candidate_topics = [t.strip() for t in custom_topics_str.split(",") if t.strip()]

# Memory cleaner in sidebar
if st.sidebar.button("🧹 Clear Server Cache / RAM"):
    st.session_state.pipeline_cache = {}
    clean_gpu_memory()
    st.sidebar.success("Cache and GPU memory cleared successfully.")

# Cache the loader for summarizer pipelines
@st.cache_resource(show_spinner=False)
def load_pipeline(summarizer_name: str, classifier_name: str, token: str, gemini_api_key: str) -> DocumentSummarizerPipeline:
    return DocumentSummarizerPipeline(
        summarizer_model_name=summarizer_name,
        classifier_model_name=classifier_name,
        hf_api_token=token,
        gemini_api_key=gemini_api_key
    )
# Helper to process a single document from stream
def process_single_file_stream(uploaded_file, pipeline_obj, model_option, classifier_model, hf_token, gemini_key, summary_mode, summary_length, temp, beams, penalty, fast_mode, enable_topic, enable_sentiment, enable_keywords, candidate_topics, progress_callback=None):
    filename = uploaded_file.name
    file_bytes = uploaded_file.getvalue()
    file_ext = filename.split('.')[-1].lower()
    
    # 1. Text extraction
    if progress_callback:
        progress_callback(0.05, f"Extracting text from {filename}...")
    raw_text = ""
    if file_ext == 'pdf':
        raw_text = extract_text_from_pdf(file_bytes)
    elif file_ext == 'docx':
        raw_text = extract_text_from_docx(file_bytes)
    elif file_ext == 'txt':
        raw_text = extract_text_from_txt(file_bytes)
        
    if not raw_text.strip():
        raise ValueError("Document contains no readable text.")
        
    cleaned_text = clean_text(raw_text)
    
    if gemini_key:
        # ----------------- GEMINI UNIFIED PIPELINE -----------------
        if progress_callback:
            progress_callback(0.2, "Querying Google Gemini 1.5 Flash API...")
            
        gemini_data = pipeline_obj.generate_gemini_analysis(
            cleaned_text, 
            summary_mode, 
            summary_length, 
            candidate_topics
        )
        
        if progress_callback:
            progress_callback(0.8, "Processing Gemini API response...")
            
        base_summary = gemini_data.get("summary", "")
        final_formatted_summary = gemini_data.get("formatted_summary", "")
        
        # Convert keywords format
        raw_keywords = gemini_data.get("keywords", [])
        keywords = [(k[0], float(k[1])) for k in raw_keywords if isinstance(k, list) and len(k) >= 2]
        topics = gemini_data.get("topics", {})
        raw_senti = gemini_data.get("sentiment", {})
        sentiment = raw_senti if raw_senti else {"Positive": 0.33, "Negative": 0.33, "Neutral": 0.34}
        num_chunks = 1
    else:
        # ----------------- LOCAL / REMOTE HUGGING FACE PIPELINE -----------------
        if progress_callback:
            progress_callback(0.15, "Preparing Hugging Face models...")
            
        # Define a sub-callback to capture generate_summary updates
        def sum_cb(frac, msg):
            if progress_callback:
                # Map 0.0-1.0 of generate_summary to 0.2-0.75 range
                progress_callback(0.2 + frac * 0.55, f"Summarizing: {msg}")
                
        base_summary, num_chunks = pipeline_obj.generate_summary(
            cleaned_text, 
            length_setting=summary_length, 
            temperature=temp,
            num_beams=beams,
            length_penalty=penalty,
            progress_callback=sum_cb
        )
        
        if progress_callback:
            progress_callback(0.78, "Formatting summary structure...")
        # Format Summary Mode
        final_formatted_summary = pipeline_obj.format_summary_mode(cleaned_text, base_summary, summary_mode)
        
        # Choose analysis source based on Fast Extraction Mode
        analysis_source = base_summary if fast_mode else cleaned_text
        
        # 3. Topic classification
        if enable_topic:
            if progress_callback:
                progress_callback(0.82, "Classifying topics...")
            topics = pipeline_obj.classify_topics(analysis_source, candidate_topics)
        else:
            topics = {}
            
        # 4. Sentiment analysis
        if enable_sentiment:
            if progress_callback:
                progress_callback(0.88, "Analyzing sentiment...")
            sentiment = pipeline_obj.analyze_sentiment(analysis_source)
        else:
            sentiment = {}
            
        # 5. Keyword extraction
        if enable_keywords:
            if progress_callback:
                progress_callback(0.92, "Extracting keywords...")
            keywords = pipeline_obj.extract_keywords(analysis_source, top_n=10)
        else:
            keywords = []
            
    # Calculate document metrics
    if progress_callback:
        progress_callback(0.96, "Calculating document metrics...")
    stats = get_document_stats(cleaned_text, base_summary)
    stats["num_chunks_processed"] = num_chunks
    
    # Generate download bytes
    if progress_callback:
        progress_callback(0.98, "Generating Word & PDF reports...")
    pdf_bytes = get_pdf_download_bytes(filename, final_formatted_summary, stats, keywords)
    docx_bytes = get_docx_download_bytes(filename, final_formatted_summary, stats, keywords)
    
    if progress_callback:
        progress_callback(1.0, "Completed!")
        
    return {
        "extracted_text": cleaned_text,
        "summary": final_formatted_summary,
        "stats": stats,
        "keywords": keywords,
        "topics": topics,
        "sentiment": sentiment,
        "pdf_bytes": pdf_bytes,
        "docx_bytes": docx_bytes,
        "file_size_kb": len(file_bytes) / 1024
    }

# Main Page Split
col_uploader, col_intro = st.columns([2, 1])

with col_uploader:
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT document(s) (Up to 50MB per file)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Drag and drop or select files. Supported formats: Adobe PDF (.pdf), MS Word (.docx), Plain Text (.txt)"
    )

with col_intro:
    st.markdown("""
    #### 📋 How to Use:
    1. **Upload** one or more PDF, DOCX, or text files.
    2. **Configure** parameters and select remote API or local modes in the sidebar.
    3. Click **Process & Analyze Document(s)**.
    4. Explore summary tabs, inspect files individually, and download combined reports!
    
    ⚡ _Running on **{}** hardware acceleration._
    """.format(get_device_str().upper()))

if uploaded_files:
    # Process Button
    if st.button("🚀 Process & Analyze Document(s)"):
        st.session_state.processed_docs = {}
        st.session_state.batch_zip = b""
        
        pipeline_obj = load_pipeline(model_option, classifier_model, hf_token, gemini_key)
        is_remote = bool(gemini_key or hf_token)
        
        # Start timer
        start_time = time.time()
        
        # Display status placeholders and progress indicators
        st.write("### ⏳ AI Document Processing Queue")
        main_progress_bar = st.progress(0.0)
        main_status_text = st.empty()
        
        status_placeholders = {}
        for uf in uploaded_files:
            col_name, col_status = st.columns([2, 3])
            with col_name:
                st.write(f"📄 **{uf.name}**")
            with col_status:
                status_placeholders[uf.name] = st.empty()
                status_placeholders[uf.name].info("Queued...")
        
        # Parallel thread executor for remote APIs, sequential for local execution to prevent OOM
        max_workers = 4 if is_remote else 1
        
        results = {}
        total_files = len(uploaded_files)
        
        if not is_remote:
            # Sequential execution in the main thread (fully safe to update UI)
            for idx, uf in enumerate(uploaded_files):
                base_pct = idx / total_files
                
                def seq_cb(frac, msg):
                    file_pct = base_pct + (frac / total_files)
                    main_progress_bar.progress(min(1.0, file_pct))
                    pct_val = int(file_pct * 100)
                    main_status_text.markdown(f"**Progress: {pct_val}%** (File {idx+1}/{total_files}: {msg})")
                    if frac < 1.0:
                        status_placeholders[uf.name].warning(msg)
                    else:
                        status_placeholders[uf.name].success("Completed!")
                
                try:
                    res_data = process_single_file_stream(
                        uf, pipeline_obj, model_option, classifier_model, hf_token, gemini_key,
                        summary_mode, summary_length, temp, beams, penalty, fast_mode,
                        enable_topic, enable_sentiment, enable_keywords, candidate_topics,
                        progress_callback=seq_cb
                    )
                    results[uf.name] = res_data
                except Exception as ex:
                    status_placeholders[uf.name].error(f"Failed: {str(ex)}")
            
            main_progress_bar.progress(1.0)
            main_status_text.markdown("**Progress: 100%** (Batch processing completed!)")
        else:
            # Remote API mode - run thread pool but update UI only from parent thread on completion
            from concurrent.futures import ThreadPoolExecutor
            
            progress_dict = {uf.name: {"pct": 0.0, "msg": "Queued..."} for uf in uploaded_files}
            
            def worker_fn(uf):
                def worker_cb(frac, msg):
                    progress_dict[uf.name] = {"pct": frac, "msg": msg}
                
                try:
                    res_data = process_single_file_stream(
                        uf, pipeline_obj, model_option, classifier_model, hf_token, gemini_key,
                        summary_mode, summary_length, temp, beams, penalty, fast_mode,
                        enable_topic, enable_sentiment, enable_keywords, candidate_topics,
                        progress_callback=worker_cb
                    )
                    progress_dict[uf.name] = {"pct": 1.0, "msg": "Completed!"}
                    return uf.name, res_data, None
                except Exception as ex:
                    progress_dict[uf.name] = {"pct": 1.0, "msg": f"Failed: {str(ex)}"}
                    return uf.name, None, str(ex)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(worker_fn, uf): uf for uf in uploaded_files}
                
                while not all(f.done() for f in futures):
                    # Compute aggregate progress
                    total_pct = sum(progress_dict[uf.name]["pct"] for uf in uploaded_files) / total_files
                    main_progress_bar.progress(min(1.0, total_pct))
                    
                    # Update status message for each file
                    for uf in uploaded_files:
                        msg = progress_dict[uf.name]["msg"]
                        pct = progress_dict[uf.name]["pct"]
                        if pct >= 1.0:
                            if "Failed" in msg:
                                status_placeholders[uf.name].error(msg)
                            else:
                                status_placeholders[uf.name].success(msg)
                        else:
                            status_placeholders[uf.name].warning(msg)
                            
                    pct_complete = int(total_pct * 100)
                    main_status_text.markdown(f"**Progress: {pct_complete}%** (Processing documents in parallel... {pct_complete}% done)")
                    time.sleep(0.1)
                
                # Final check to set correct states for everyone
                for uf in uploaded_files:
                    msg = progress_dict[uf.name]["msg"]
                    if "Failed" in msg:
                        status_placeholders[uf.name].error(msg)
                    else:
                        status_placeholders[uf.name].success("Completed!")
                
                main_progress_bar.progress(1.0)
                main_status_text.markdown("**Progress: 100%** (Batch processing completed!)")
                
                # Gather all results
                for future in futures:
                    fname, res_data, err = future.result()
                    if res_data:
                        results[fname] = res_data
                            
        # Store in session state
        st.session_state.processed_docs = results
        if results:
            st.session_state.selected_doc_name = list(results.keys())[0]
            st.session_state.is_processed = True
            
            # Generate zip archive of summaries
            import zipfile
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                for fname, data in results.items():
                    base_name = os.path.splitext(fname)[0]
                    zip_file.writestr(f"summary_{base_name}.txt", data["summary"])
                    if data["docx_bytes"]:
                        zip_file.writestr(f"summary_{base_name}.docx", data["docx_bytes"])
                    if data["pdf_bytes"]:
                        zip_file.writestr(f"summary_{base_name}.pdf", data["pdf_bytes"])
            st.session_state.batch_zip = zip_buffer.getvalue()
            
            # Display complete message
            total_latency = round(time.time() - start_time, 2)
            st.success(f"Processed {len(results)} / {len(uploaded_files)} files successfully in {total_latency} seconds!")
            time.sleep(1.0)
            st.rerun()

# Render Results
if st.session_state.is_processed and st.session_state.processed_docs:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # If multiple files are processed, show a batch overview section
    if len(st.session_state.processed_docs) > 1:
        st.subheader("📦 Multi-Document Batch Overview")
        # Draw batch metric grid
        total_docs = len(st.session_state.processed_docs)
        total_source_words = sum(doc["stats"].get("word_count", 0) for doc in st.session_state.processed_docs.values())
        total_summary_words = sum(doc["stats"].get("summary_word_count", 0) for doc in st.session_state.processed_docs.values())
        avg_reduction = round(((total_source_words - total_summary_words) / total_source_words * 100), 1) if total_source_words > 0 else 0.0
        
        batch_col1, batch_col2, batch_col3, batch_col4 = st.columns(4)
        with batch_col1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-val">{total_docs}</div>
                <div class="kpi-lbl">Total Documents</div>
            </div>
            """, unsafe_allow_html=True)
        with batch_col2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-val">{total_source_words:,}</div>
                <div class="kpi-lbl">Aggregate Words</div>
            </div>
            """, unsafe_allow_html=True)
        with batch_col3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-val">{avg_reduction}%</div>
                <div class="kpi-lbl">Average Reduction</div>
            </div>
            """, unsafe_allow_html=True)
        with batch_col4:
            st.write("")
            st.download_button(
                label="📦 Download All Reports (ZIP)",
                data=st.session_state.batch_zip,
                file_name="summarizer_batch_reports.zip",
                mime="application/zip",
                use_container_width=True
            )
            
        # Draw summary table of all processed documents
        batch_records = []
        for fname, data in st.session_state.processed_docs.items():
            batch_records.append({
                "Document Name": fname,
                "Size (KB)": round(data.get("file_size_kb", 0), 1),
                "Original Words": data["stats"].get("word_count", 0),
                "Summary Words": data["stats"].get("summary_word_count", 0),
                "Reduction": f"{data['stats'].get('reduction_percentage', 0)}%"
            })
        st.dataframe(pd.DataFrame(batch_records), use_container_width=True, hide_index=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
    # Dropdown selector to inspect a specific document
    if len(st.session_state.processed_docs) > 1:
        st.subheader("🔍 Inspect Detailed Document Report")
        selected_doc = st.selectbox(
            "Choose a document from the processed batch to inspect details:",
            options=list(st.session_state.processed_docs.keys()),
            index=list(st.session_state.processed_docs.keys()).index(st.session_state.selected_doc_name) if st.session_state.selected_doc_name in st.session_state.processed_docs else 0
        )
        st.session_state.selected_doc_name = selected_doc
    else:
        st.session_state.selected_doc_name = list(st.session_state.processed_docs.keys())[0]
        
    # Now pull selected document details
    current_doc_data = st.session_state.processed_docs[st.session_state.selected_doc_name]
    stats = current_doc_data["stats"]
    summary_text = current_doc_data["summary"]
    keywords = current_doc_data["keywords"]
    topics = current_doc_data["topics"]
    sentiment = current_doc_data["sentiment"]
    extracted_text = current_doc_data["extracted_text"]
    
    # Draw selected document stats
    st.markdown(f"### 📊 Report for: **{st.session_state.selected_doc_name}**")
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-val">{stats.get('word_count', 0):,}</div>
            <div class="kpi-lbl">Source Words</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-val">{stats.get('summary_word_count', 0):,}</div>
            <div class="kpi-lbl">Summary Words</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-val">{stats.get('reduction_percentage', 0)}%</div>
            <div class="kpi-lbl">Reduction</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-val">{stats.get('reading_time_mins', 0)}m</div>
            <div class="kpi-lbl">Est. Reading Time</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visualisation and export tabs
    tab_summary, tab_charts, tab_metadata = st.tabs([
        "📝 Summarized Output Report",
        "📈 Interactive Visualizations",
        "📂 Source Details & Metadata"
    ])
    
    with tab_summary:
        col_summary_content, col_actions = st.columns([3, 1])
        
        with col_summary_content:
            st.markdown(f"### 🎯 Synthesized {summary_mode}")
            
            # Collapsible editor for those who want to modify/edit the summary
            with st.expander("✍️ Edit Summary Text (Optional)"):
                edited_summary = st.text_area(
                    "Modify the summary text below to customize your exported report:",
                    value=summary_text,
                    height=300,
                    key=f"editor_{st.session_state.selected_doc_name}",
                    help="You can manually edit this text area. The download buttons on the right will export your edited version."
                )
            
            st.markdown("#### 📖 Document Summary")
            st.markdown(edited_summary)
            
            # If summary has been edited, update the dictionary and cache bytes
            if current_doc_data["summary"] != edited_summary:
                current_doc_data["summary"] = edited_summary
                current_doc_data["pdf_bytes"] = get_pdf_download_bytes(
                    st.session_state.selected_doc_name, 
                    edited_summary, 
                    stats, 
                    keywords
                )
                current_doc_data["docx_bytes"] = get_docx_download_bytes(
                    st.session_state.selected_doc_name, 
                    edited_summary, 
                    stats, 
                    keywords
                )
                # Re-generate batch ZIP buffer
                import zipfile
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                    for fname, data in st.session_state.processed_docs.items():
                        base_name = os.path.splitext(fname)[0]
                        zip_file.writestr(f"summary_{base_name}.txt", data["summary"])
                        if data["docx_bytes"]:
                            zip_file.writestr(f"summary_{base_name}.docx", data["docx_bytes"])
                        if data["pdf_bytes"]:
                            zip_file.writestr(f"summary_{base_name}.pdf", data["pdf_bytes"])
                st.session_state.batch_zip = zip_buffer.getvalue()
            
            # Simple text finder feature
            st.markdown("---")
            search_query = st.text_input("🔍 Search phrase in summary:", value="", key=f"search_{st.session_state.selected_doc_name}", placeholder="Type keyword to find...")
            if search_query:
                import re
                matches = len(re.findall(re.escape(search_query), edited_summary, re.IGNORECASE))
                if matches > 0:
                    st.success(f"Found **{matches}** matches of '{search_query}' in the summary.")
                else:
                    st.info(f"No matches for '{search_query}'.")
                    
        with col_actions:
            st.markdown("### 📥 Export Summarized Report")
            
            st.download_button(
                label="📄 Download Summary (.txt)",
                data=edited_summary,
                file_name=f"summary_{os.path.splitext(st.session_state.selected_doc_name)[0]}.txt",
                mime="text/plain",
                key=f"download_txt_{st.session_state.selected_doc_name}"
            )
            
            st.download_button(
                label="📘 Download Summary (.docx)",
                data=current_doc_data["docx_bytes"],
                file_name=f"summary_{os.path.splitext(st.session_state.selected_doc_name)[0]}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key=f"download_docx_{st.session_state.selected_doc_name}"
            )
            
            st.download_button(
                label="📕 Download Summary (.pdf)",
                data=current_doc_data["pdf_bytes"],
                file_name=f"summary_{os.path.splitext(st.session_state.selected_doc_name)[0]}.pdf",
                mime="application/pdf",
                key=f"download_pdf_{st.session_state.selected_doc_name}"
            )
            
            st.info("""
            ℹ️ **Report Layout Includes:**
            - Executive Summary content
            - Structured section headers
            - Document statistics
            - Extracted keywords
            """)
            
    with tab_charts:
        _ensure_plotly()
        st.markdown("### 📊 Document Analytics & Visualizations")
        
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)
        
        # 1. Keywords bar chart
        with row1_col1:
            if enable_keywords and keywords:
                kw_df = pd.DataFrame(keywords, columns=["Keyword", "Relevance Score"])
                kw_df = kw_df.sort_values(by="Relevance Score", ascending=True)
                fig_kw = px.bar(
                    kw_df,
                    x="Relevance Score",
                    y="Keyword",
                    orientation="h",
                    title="🔑 KeyBERT Top Key Phrases Relevance",
                    color="Relevance Score",
                    color_continuous_scale="Blugrn"
                )
                fig_kw.update_layout(
                    height=350, 
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#94A3B8'
                )
                st.plotly_chart(fig_kw, use_container_width=True, key=f"chart_kw_{st.session_state.selected_doc_name}")
            else:
                st.info("💡 Keyword Extraction is disabled or not available.")
                
        # 2. Topic classification bar chart
        with row1_col2:
            if enable_topic and topics:
                topic_data = sorted(topics.items(), key=lambda x: x[1], reverse=False)
                topic_df = pd.DataFrame(topic_data, columns=["Topic", "Confidence Score"])
                fig_topic = px.bar(
                    topic_df,
                    x="Confidence Score",
                    y="Topic",
                    orientation="h",
                    title="🏷️ Zero-Shot Topic Classification Confidence",
                    color="Confidence Score",
                    color_continuous_scale="Purp"
                )
                fig_topic.update_layout(
                    height=350, 
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#94A3B8'
                )
                st.plotly_chart(fig_topic, use_container_width=True, key=f"chart_topic_{st.session_state.selected_doc_name}")
            else:
                st.info("💡 Topic Detection is disabled or not available.")
                
        # 3. Sentiment pie chart
        with row2_col1:
            if enable_sentiment and sentiment:
                senti_data = list(sentiment.items())
                senti_df = pd.DataFrame(senti_data, columns=["Sentiment", "Confidence"])
                fig_senti = px.pie(
                    senti_df,
                    values="Confidence",
                    names="Sentiment",
                    title="🎭 Document Sentiment Distribution",
                    color="Sentiment",
                    color_discrete_map={"Positive": "#10B981", "Negative": "#EF4444", "Neutral": "#64748B"}
                )
                fig_senti.update_traces(textposition='inside', textinfo='percent+label')
                fig_senti.update_layout(
                    height=350, 
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#94A3B8'
                )
                st.plotly_chart(fig_senti, use_container_width=True, key=f"chart_senti_{st.session_state.selected_doc_name}")
            else:
                st.info("💡 Sentiment Analysis is disabled or not available.")
                
        # 4. Compression Ratio Gauge Chart
        with row2_col2:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = stats.get('reduction_percentage', 0),
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "📉 Summary Reduction Percentage (%)", 'font': {'size': 16, 'color': '#94A3B8'}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#818CF8"},
                    'bar': {'color': "#818CF8"},
                    'bgcolor': "#131A2C",
                    'borderwidth': 1,
                    'bordercolor': "#1E293B",
                    'steps': [
                        {'range': [0, 50], 'color': '#131A2C'},
                        {'range': [50, 80], 'color': '#1E1B4B'},
                        {'range': [80, 100], 'color': '#312E81'}
                    ],
                }
            ))
            fig_gauge.update_layout(
                height=350, 
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#94A3B8'
            )
            st.plotly_chart(fig_gauge, use_container_width=True, key=f"chart_gauge_{st.session_state.selected_doc_name}")
            
    with tab_metadata:
        col_meta1, col_meta2 = st.columns(2)
        
        with col_meta1:
            st.markdown("### 📂 Source Document Metrics")
            st.write(f"**Filename:** {st.session_state.selected_doc_name}")
            st.write(f"**Filesize:** {round(current_doc_data.get('file_size_kb', 0), 2)} KB")
            st.write(f"**Total Paragraphs / Items:** {len(extracted_text.split(chr(10) + chr(10)))}")
            st.write(f"**Hierarchical Chunks Processed:** {stats.get('num_chunks_processed', 1)}")
            
            st.markdown("### 🤖 Engine Specs")
            st.write(f"**Transformer Summarizer:** {model_option}")
            st.write(f"**Topic Classifier:** {classifier_model}")
            st.write(f"**Device Accelerant:** {get_device_str().upper()}")
            
        with col_meta2:
            st.markdown("### 📄 Extracted Document Text (Preview)")
            st.text_area(
                "Original text extract:",
                value=extracted_text[:4000] + ("\n\n[... Truncated for preview ...]" if len(extracted_text) > 4000 else ""),
                height=300,
                key=f"text_preview_{st.session_state.selected_doc_name}",
                disabled=True
            )
else:
    # Showcase information when no document is processed
    st.info("👋 Upload document file(s) in the left column and click 'Process & Analyze Document(s)' to begin!")
    
    # Feature Showcase Grid
    st.markdown("### 🌟 Platform Capabilities")
    show_col1, show_col2, show_col3 = st.columns(3)
    
    with show_col1:
        st.markdown("""
        #### 🗺️ Hierarchical Chunking
        Safely processes document inputs of arbitrary length (50+ pages) by recursively splitting, summarizing individual text segments, and synthesising them.
        """)
        
    with show_col2:
        st.markdown("""
        #### 🏷️ Topic & Sentiment Models
        Runs zero-shot text classification pipelines to map content categories and measure emotional valence without fine-tuning overhead.
        """)
        
    with show_col3:
        st.markdown("""
        #### 📊 Keyphrase Spotting
        Utilizes KeyBERT (Sentence-BERT embedder) to index primary conceptual keywords and semantic terms from the text corpus.
        """)
