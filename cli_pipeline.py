import os
import argparse
import time
import json
import csv
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_txt,
    clean_text,
    get_document_stats,
    get_pdf_download_bytes,
    get_docx_download_bytes
)
from summarizer import DocumentSummarizerPipeline, clean_gpu_memory

def process_single_file(file_path: str, output_dir: str, pipeline: DocumentSummarizerPipeline, 
                        args: argparse.Namespace) -> Dict[str, Any]:
    """
    Processes a single document: extracts text, generates summary, runs analytics, and exports reports.
    """
    filename = os.path.basename(file_path)
    file_ext = filename.split('.')[-1].lower()
    print(f"[START] Processing {filename}...")
    
    start_time = time.time()
    result_meta = {
        "filename": filename,
        "status": "Success",
        "error": "",
        "word_count": 0,
        "summary_word_count": 0,
        "reduction_percentage": 0.0,
        "processing_time_sec": 0.0
    }
    
    try:
        # Read file bytes
        with open(file_path, "rb") as f:
            file_bytes = f.read()
            
        # Extract Text
        if file_ext == 'pdf':
            raw_text = extract_text_from_pdf(file_bytes)
        elif file_ext == 'docx':
            raw_text = extract_text_from_docx(file_bytes)
        elif file_ext == 'txt':
            raw_text = extract_text_from_txt(file_bytes)
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}")
            
        if not raw_text.strip():
            raise ValueError("Document contains no readable text.")
            
        cleaned_text = clean_text(raw_text)
        
        # Analyze Document
        if args.gemini_key:
            # Remote Gemini API
            candidate_topics = [t.strip() for t in args.topics.split(",") if t.strip()]
            analysis = pipeline.generate_gemini_analysis(
                cleaned_text,
                args.mode,
                args.length,
                candidate_topics
            )
            base_summary = analysis.get("summary", "")
            summary = analysis.get("formatted_summary", "")
            topics = analysis.get("topics", {})
            sentiment = analysis.get("sentiment", {})
            raw_keywords = analysis.get("keywords", [])
            keywords = [(k[0], float(k[1])) for k in raw_keywords if isinstance(k, list) and len(k) >= 2]
        else:
            # Local / Remote HF API
            base_summary, num_chunks = pipeline.generate_summary(
                cleaned_text,
                length_setting=args.length,
                temperature=args.temperature,
                num_beams=args.beams,
                length_penalty=args.length_penalty
            )
            summary = pipeline.format_summary_mode(cleaned_text, base_summary, args.mode)
            
            analysis_source = base_summary if args.fast else cleaned_text
            
            # Topic classification
            candidate_topics = [t.strip() for t in args.topics.split(",") if t.strip()]
            topics = pipeline.classify_topics(analysis_source, candidate_topics)
            
            # Sentiment analysis
            sentiment = pipeline.analyze_sentiment(analysis_source)
            
            # Keywords extraction
            keywords = pipeline.extract_keywords(analysis_source, top_n=10)
            
        # Generate stats
        stats = get_document_stats(cleaned_text, base_summary)
        processing_time = round(time.time() - start_time, 2)
        
        # Update metadata results
        result_meta.update({
            "word_count": stats.get("word_count", 0),
            "summary_word_count": stats.get("summary_word_count", 0),
            "reduction_percentage": stats.get("reduction_percentage", 0.0),
            "processing_time_sec": processing_time
        })
        
        # Output paths
        base_output_name = os.path.splitext(filename)[0]
        
        # Save TXT
        txt_path = os.path.join(output_dir, f"summary_{base_output_name}.txt")
        with open(txt_path, "w", encoding="utf-8") as out_txt:
            out_txt.write(summary)
            
        # Save DOCX
        docx_bytes = get_docx_download_bytes(filename, summary, stats, keywords)
        if docx_bytes:
            docx_path = os.path.join(output_dir, f"summary_{base_output_name}.docx")
            with open(docx_path, "wb") as out_docx:
                out_docx.write(docx_bytes)
                
        # Save PDF
        pdf_bytes = get_pdf_download_bytes(filename, summary, stats, keywords)
        if pdf_bytes:
            pdf_path = os.path.join(output_dir, f"summary_{base_output_name}.pdf")
            with open(pdf_path, "wb") as out_pdf:
                out_pdf.write(pdf_bytes)
                
        # Save detailed JSON analytics metadata
        json_path = os.path.join(output_dir, f"analytics_{base_output_name}.json")
        analytics_data = {
            "document_name": filename,
            "statistics": stats,
            "processing_time_sec": processing_time,
            "extracted_keywords": keywords,
            "topics": topics,
            "sentiment": sentiment,
            "summary_text": summary
        }
        with open(json_path, "w", encoding="utf-8") as out_json:
            json.dump(analytics_data, out_json, indent=4)
            
        print(f"[SUCCESS] Processed {filename} in {processing_time}s")
        
    except Exception as e:
        processing_time = round(time.time() - start_time, 2)
        result_meta.update({
            "status": "Failed",
            "error": str(e),
            "processing_time_sec": processing_time
        })
        print(f"[FAILED] Error processing {filename}: {str(e)}")
        
    return result_meta

def main():
    parser = argparse.ArgumentParser(description="AI Batch Document Summarization & Analytics CLI Pipeline")
    
    # Paths
    parser.add_argument("--input", "-i", required=True, help="Path to input directory containing documents")
    parser.add_argument("--output", "-o", required=True, help="Path to output directory to save reports")
    
    # API configuration
    parser.add_argument("--hf-token", help="Hugging Face User Access Token (for remote Inference API)")
    parser.add_argument("--gemini-key", help="Google Gemini API Key (for remote Gemini 1.5 Flash API)")
    
    # Model parameters
    parser.add_argument("--model", "-m", default="sshleifer/distilbart-cnn-6-6", 
                        help="Hugging Face summarizer model name (default: sshleifer/distilbart-cnn-6-6, fastest CPU option)")
    parser.add_argument("--classifier-model", default="valhalla/distilbart-mnli-12-6",
                        help="Hugging Face zero-shot classifier model name")
    
    # Text Generation / Summary styles
    parser.add_argument("--mode", default="Executive Summary", 
                        choices=["Executive Summary", "Bullet Point Summary", "Detailed Summary", 
                                 "Meeting Notes Summary", "Research Paper Summary", "Key Insights Summary"],
                        help="Summary Structuring Style Mode")
    parser.add_argument("--length", default="medium", choices=["short", "medium", "long"],
                        help="Target length bounds")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--beams", type=int, default=4, help="Beam search count")
    parser.add_argument("--length-penalty", type=float, default=2.0, help="Generation length penalty")
    
    # Performance toggles
    parser.add_argument("--fast", type=bool, default=True, help="Run analytics on generated summary for 10x speedup")
    parser.add_argument("--concurrency", "-c", type=int, default=4, help="Max parallel API workers")
    parser.add_argument("--topics", default="Technology, Finance, Healthcare, Education, Research, Business, Legal, Government",
                        help="Comma-separated candidate topics")
    parser.add_argument("--onnx", action="store_true", default=True,
                        help="Enable ONNX Runtime acceleration for CPU (default: enabled, falls back to PyTorch if not installed)")
    parser.add_argument("--no-onnx", action="store_true", default=False,
                        help="Disable ONNX Runtime acceleration (force PyTorch)")
    parser.add_argument("--turbo", action="store_true", default=False,
                        help="Maximum speed mode: greedy decoding (1 beam) + aggressive compression")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input directory '{args.input}' does not exist.")
        return
        
    os.makedirs(args.output, exist_ok=True)
    
    # Gather files
    supported_extensions = ('.pdf', '.docx', '.txt')
    files_to_process = [
        os.path.join(args.input, f) for f in os.listdir(args.input)
        if f.lower().endswith(supported_extensions) and os.path.isfile(os.path.join(args.input, f))
    ]
    
    if not files_to_process:
        print(f"No supported documents (.pdf, .docx, .txt) found in {args.input}")
        return
        
    print(f"Found {len(files_to_process)} documents to process.")
    
    # Apply turbo mode overrides
    if args.turbo:
        args.beams = 1
        print("Turbo mode enabled: greedy decoding (1 beam) for maximum speed.")
    
    use_onnx = args.onnx and not args.no_onnx
    
    # Instantiate Pipeline
    print(f"Initializing AI Pipeline (Model: {args.model}, ONNX: {use_onnx})...")
    pipeline = DocumentSummarizerPipeline(
        summarizer_model_name=args.model,
        classifier_model_name=args.classifier_model,
        hf_api_token=args.hf_token or "",
        gemini_api_key=args.gemini_key or "",
        use_onnx=use_onnx
    )
    
    # Determine concurrency
    # Local PyTorch runs are very CPU/GPU memory intensive. If no remote APIs are provided, 
    # run sequentially to prevent memory overflow and thermal throttling.
    is_remote = bool(args.hf_token or args.gemini_key)
    max_workers = args.concurrency if is_remote else 1
    
    if not is_remote:
        print("Note: Running in local PyTorch mode. Processing files sequentially (1 worker) to protect local system resources.")
    else:
        print(f"Running in remote API mode. Concurrency enabled with {max_workers} worker threads.")
        
    results = []
    
    start_pipeline_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_file, f, args.output, pipeline, args): f 
            for f in files_to_process
        }
        
        for future in as_completed(futures):
            f_path = futures[future]
            res = future.result()
            results.append(res)
            
    total_elapsed = round(time.time() - start_pipeline_time, 2)
    
    # Clean up memory
    clean_gpu_memory()
    
    # Write aggregated CSV batch report
    csv_report_path = os.path.join(args.output, "batch_processing_report.csv")
    csv_fields = ["filename", "status", "word_count", "summary_word_count", "reduction_percentage", "processing_time_sec", "error"]
    
    try:
        with open(csv_report_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
            writer.writeheader()
            for row in results:
                # filter row to match fields
                filtered_row = {k: row.get(k, "") for k in csv_fields}
                writer.writerow(filtered_row)
        print(f"\n[REPORT] Saved aggregated CSV summary to {csv_report_path}")
    except Exception as e:
        print(f"Failed to write CSV summary report: {e}")
        
    # Print completion summary
    successful_runs = sum(1 for r in results if r["status"] == "Success")
    failed_runs = len(results) - successful_runs
    print(f"\n==================================================")
    print(f"BATCH PROCESSING COMPLETED IN {total_elapsed} SECONDS")
    print(f"Total Files: {len(files_to_process)}")
    print(f"Success: {successful_runs}")
    print(f"Failed: {failed_runs}")
    print(f"==================================================")

if __name__ == "__main__":
    main()
