import gc
import re
import requests
import time
import os
from typing import List, Dict, Any, Tuple

# Lazily checked device variables
_DEVICE = None
_DEVICE_STR = None
_ONNX_AVAILABLE = None

def get_device():
    global _DEVICE, _DEVICE_STR
    if _DEVICE is None:
        try:
            import torch
            _DEVICE = 0 if torch.cuda.is_available() else -1
            _DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            _DEVICE = -1
            _DEVICE_STR = "cpu"
    return _DEVICE, _DEVICE_STR

def get_device_str():
    return get_device()[1]

def is_onnx_available():
    """Check if ONNX Runtime and HuggingFace Optimum are installed."""
    global _ONNX_AVAILABLE
    if _ONNX_AVAILABLE is None:
        try:
            import onnxruntime
            from optimum.onnxruntime import ORTModelForSeq2SeqLM
            _ONNX_AVAILABLE = True
        except ImportError:
            _ONNX_AVAILABLE = False
    return _ONNX_AVAILABLE

def clean_gpu_memory():
    """
    Cleans up unused GPU and CPU memory by invoking garbage collector and clearing torch cache.
    """
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

def get_sentence_splitter_regex():
    """
    Returns a regex splitter to break text into sentences based on punctuation.
    """
    # Splitting on periods, question marks, and exclamation marks followed by spaces or newlines
    return re.compile(r'(?<=[\.!\?])\s+')

def split_text_into_sentences(text: str) -> List[str]:
    """
    Splits text into sentences using regex boundary detection.
    """
    splitter = get_sentence_splitter_regex()
    raw_sentences = splitter.split(text)
    # Filter out empty entries and strip spacing
    return [s.strip() for s in raw_sentences if s.strip()]

class DocumentSummarizerPipeline:
    """
    A unified interface for document understanding:
    - Text Summarization (Hierarchical and Single Pass via Local/API Pipeline)
    - Topic Classification (Zero-Shot via Local/API NLI)
    - Sentiment Analysis (Zero-Shot via Local/API NLI)
    - Keyword/Keyphrase Extraction (KeyBERT)
    """
    
    def __init__(self, summarizer_model_name: str = "sshleifer/distilbart-cnn-6-6", 
                 classifier_model_name: str = "valhalla/distilbart-mnli-12-6",
                 hf_api_token: str = "",
                 gemini_api_key: str = "",
                 use_onnx: bool = True):
        self.summarizer_model_name = summarizer_model_name
        self.classifier_model_name = classifier_model_name
        self.hf_api_token = hf_api_token.strip()
        self.gemini_api_key = gemini_api_key.strip()
        self.use_onnx = use_onnx
        
        # Pipelines and models are initialized lazily
        self._summarizer_pipeline = None
        self._classifier_pipeline = None
        self._keybert_model = None
        self._tokenizer = None
        self._is_onnx_model = False

    def _query_hf_api(self, model_name: str, payload: dict) -> dict:
        """
        Sends requests to the remote Hugging Face Serverless Inference API.
        Handles model loading errors (503) by retrying after the suggested wait period.
        """
        api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        headers = {"Authorization": f"Bearer {self.hf_api_token}"}
        
        for attempt in range(5):
            try:
                response = requests.post(api_url, headers=headers, json=payload, timeout=40)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 503:
                    load_info = response.json()
                    estimated_time = load_info.get("estimated_time", 8)
                    print(f"Hugging Face server is spinning up '{model_name}'. Waiting {estimated_time}s (attempt {attempt+1}/5)...")
                    time.sleep(estimated_time)
                else:
                    raise ValueError(f"Hugging Face API returned status code {response.status_code}: {response.text}")
            except requests.exceptions.RequestException as e:
                if attempt == 4:
                    raise e
                time.sleep(2)
        raise ValueError(f"Hugging Face Inference API timed out waiting for '{model_name}' to load.")

    @property
    def tokenizer(self):
        """Loads and returns the tokenizer for the chosen summarizer model."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.summarizer_model_name)
        return self._tokenizer

    @property
    def summarizer(self):
        """Loads and returns the Hugging Face model object (if in local mode).
        Tries ONNX Runtime first for 2-4x CPU speedup, falls back to PyTorch."""
        if self._summarizer_pipeline is None and not self.hf_api_token:
            device, dev_str_val = get_device()
            
            # Try ONNX Runtime for massive CPU speedup
            if self.use_onnx and dev_str_val == "cpu" and is_onnx_available():
                try:
                    from optimum.onnxruntime import ORTModelForSeq2SeqLM
                    print(f"Loading ONNX-optimized model: {self.summarizer_model_name}")
                    model = ORTModelForSeq2SeqLM.from_pretrained(
                        self.summarizer_model_name,
                        export=True,
                        provider="CPUExecutionProvider"
                    )
                    self._summarizer_pipeline = model
                    self._is_onnx_model = True
                    print("ONNX Runtime model loaded successfully (2-4x CPU speedup active)")
                except Exception as e:
                    print(f"ONNX loading failed ({e}), falling back to PyTorch...")
                    self._is_onnx_model = False
            
            # Fallback: standard PyTorch
            if self._summarizer_pipeline is None:
                from transformers import AutoModelForSeq2SeqLM
                model = AutoModelForSeq2SeqLM.from_pretrained(self.summarizer_model_name)
                target_device = "cuda" if device >= 0 else "cpu"
                self._summarizer_pipeline = model.to(target_device)
                self._is_onnx_model = False
        return self._summarizer_pipeline

    @property
    def classifier(self):
        """Loads and returns the Hugging Face zero-shot classification pipeline (if in local mode)."""
        if self._classifier_pipeline is None and not self.hf_api_token:
            from transformers import pipeline
            device, _ = get_device()
            self._classifier_pipeline = pipeline(
                "zero-shot-classification",
                model=self.classifier_model_name,
                device=device
            )
        return self._classifier_pipeline

    @property
    def keybert(self):
        """Loads and returns the KeyBERT model."""
        if self._keybert_model is None:
            from keybert import KeyBERT
            # keybert will load all-MiniLM-L6-v2 by default
            self._keybert_model = KeyBERT()
        return self._keybert_model

    def count_tokens(self, text: str) -> int:
        """
        Calculates the token count of a given string using the model's tokenizer.
        """
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            return self._fast_token_estimate(text)

    @staticmethod
    def _fast_token_estimate(text: str) -> int:
        """
        Ultra-fast token count estimate without loading any model.
        English text averages ~1.3 tokens per whitespace-delimited word.
        Used for chunking and size decisions to avoid slow tokenizer calls.
        """
        return max(1, int(len(text.split()) * 1.3))

    def extractive_compress(self, text: str, max_sentences: int = 150,
                            progress_callback=None) -> str:
        """
        Uses TF-IDF sentence scoring to extract the most informative sentences
        from a massive document. This reduces hundreds of pages down to the
        key ~150 sentences BEFORE running the expensive abstractive transformer.
        
        Time complexity: O(n) — runs in <2 seconds even for 500K-word documents.
        """
        sentences = split_text_into_sentences(text)
        n_sentences = len(sentences)
        
        if n_sentences <= max_sentences:
            return text
        
        if progress_callback:
            progress_callback(0.08, f"Scoring {n_sentences:,} sentences via TF-IDF extractive ranking...")
        
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            import numpy as np
            
            # Build TF-IDF matrix over all sentences
            vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=5000,
                max_df=0.95,   # Ignore terms in >95% of sentences (too common)
                min_df=2       # Ignore terms appearing only once
            )
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Score each sentence by its total TF-IDF weight
            scores = tfidf_matrix.sum(axis=1).A1
            
            # Select top sentences, preserving original document order
            top_indices = np.argsort(scores)[-max_sentences:]
            top_indices = sorted(top_indices)
            
            compressed = " ".join([sentences[i] for i in top_indices])
            
            if progress_callback:
                ratio = round(len(compressed.split()) / max(1, len(text.split())) * 100, 1)
                progress_callback(0.14, f"Extractive compression: kept {max_sentences}/{n_sentences:,} sentences ({ratio}% of words)")
            
            return compressed
            
        except Exception as e:
            # Fallback: uniform sampling if TF-IDF fails
            print(f"TF-IDF compression failed, using uniform sampling: {e}")
            step = n_sentences / max_sentences
            sampled = [sentences[int(i * step)] for i in range(max_sentences)]
            return " ".join(sampled)

    def chunk_document(self, text: str, max_chunk_tokens: int = 1024) -> List[str]:
        """
        Chunks the document into sections of at most max_chunk_tokens.
        Uses fast word-based token estimation for speed (avoids loading tokenizer
        for every sentence, which was a major bottleneck for large documents).
        Preserves sentence boundaries.
        """
        sentences = split_text_into_sentences(text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self._fast_token_estimate(sentence)
            # If a single sentence exceeds the budget, split it by words (fallback)
            if sentence_tokens > max_chunk_tokens:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                
                # Split large sentence by words
                words = sentence.split()
                temp_chunk = []
                temp_tokens = 0
                for word in words:
                    word_tokens = max(1, int(len(word) / 3))  # fast per-word estimate
                    if temp_tokens + word_tokens > max_chunk_tokens:
                        chunks.append(" ".join(temp_chunk))
                        temp_chunk = [word]
                        temp_tokens = word_tokens
                    else:
                        temp_chunk.append(word)
                        temp_tokens += word_tokens
                if temp_chunk:
                    chunks.append(" ".join(temp_chunk))
                continue
            
            if current_tokens + sentence_tokens > max_chunk_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def _get_length_bounds(self, length_setting: str, chunk_tokens: int, model_max_len: int = 1024) -> Tuple[int, int]:
        """
        Gets min/max length parameters based on requested summarization length and input token count.
        Bounds are increased vs. original to produce more comprehensive summaries.
        """
        # Adapt length boundaries relative to chunk size
        factor = min(1.0, chunk_tokens / float(model_max_len))
        
        if length_setting == "short":
            max_len = int(120 * factor)
            min_len = int(40 * factor)
        elif length_setting == "long":
            max_len = int(512 * factor)
            min_len = int(180 * factor)
        else:  # medium
            max_len = int(300 * factor)
            min_len = int(100 * factor)
            
        # Ensure parameters are positive and valid
        min_len = max(15, min_len)
        max_len = max(min_len + 15, max_len)
        return min_len, max_len

    def summarize_chunk(self, chunk_text: str, length_setting: str, 
                        temperature: float = 0.7, num_beams: int = 4, 
                        length_penalty: float = 2.0) -> str:
        """
        Summarizes a single chunk using the transformer model (local direct generate or remote API).
        Uses torch.inference_mode() to disable autograd for ~15-20% speedup.
        """
        token_count = self.count_tokens(chunk_text)
        if token_count < 40:
            # Too short to summarize, return as is
            return chunk_text
            
        # Determine max length dynamically
        model_max_len = getattr(self.tokenizer, "model_max_length", 1024)
        if not isinstance(model_max_len, int) or model_max_len > 4096 or model_max_len <= 0:
            model_max_len = 1024

        min_len, max_len = self._get_length_bounds(length_setting, token_count, model_max_len)
        
        try:
            # Some models like Flan-T5 benefit from a prefix instruction
            inputs = chunk_text
            if "flan-t5" in self.summarizer_model_name.lower():
                inputs = f"summarize: {chunk_text}"
                
            if self.hf_api_token:
                params = {
                    "max_length": max_len,
                    "min_length": min_len,
                    "num_beams": num_beams,
                    "length_penalty": length_penalty
                }
                if num_beams == 1 and temperature > 0.1:
                    params["temperature"] = temperature
                payload = {
                    "inputs": inputs,
                    "parameters": params
                }
                res = self._query_hf_api(self.summarizer_model_name, payload)
                return res[0]["summary_text"].strip()
            else:
                device = get_device_str()
                # ONNX models handle device internally, PyTorch needs explicit .to(device)
                if self._is_onnx_model:
                    model_inputs = self.tokenizer(inputs, max_length=model_max_len, truncation=True, return_tensors="pt")
                else:
                    model_inputs = self.tokenizer(inputs, max_length=model_max_len, truncation=True, return_tensors="pt").to(device)
                
                do_sample = False
                if num_beams == 1 and temperature > 0.1:
                    do_sample = True
                
                # Use inference_mode for ~15-20% speedup (disables autograd tracking)
                import torch
                with torch.inference_mode():
                    summary_ids = self.summarizer.generate(
                        model_inputs["input_ids"],
                        max_length=max_len,
                        min_length=min_len,
                        do_sample=do_sample,
                        temperature=temperature if do_sample else None,
                        num_beams=num_beams,
                        length_penalty=length_penalty,
                        early_stopping=True if num_beams > 1 else False,
                        no_repeat_ngram_size=3
                    )
                return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
        except Exception as e:
            # Fallback if summarization fails
            print(f"Error in summarizing chunk: {str(e)}")
            return chunk_text

    def _summarize_batch(self, chunks: List[str], length_setting: str,
                         temperature: float = 0.7, num_beams: int = 4,
                         length_penalty: float = 2.0, batch_size: int = 3) -> List[str]:
        """
        Batch-summarize multiple chunks at once for better CPU utilization.
        Falls back to sequential processing if batch inference fails.
        """
        if self.hf_api_token:
            # API mode: process sequentially (API handles its own batching)
            return [self.summarize_chunk(c, length_setting, temperature, num_beams, length_penalty) for c in chunks]
        
        model_max_len = getattr(self.tokenizer, "model_max_length", 1024)
        if not isinstance(model_max_len, int) or model_max_len > 4096 or model_max_len <= 0:
            model_max_len = 1024
        
        results = []
        import torch
        
        for batch_start in range(0, len(chunks), batch_size):
            batch = chunks[batch_start:batch_start + batch_size]
            
            # Filter out chunks too short to summarize
            processable = []
            passthrough_map = {}  # index -> original text (for short chunks)
            for i, chunk in enumerate(batch):
                token_est = self._fast_token_estimate(chunk)
                if token_est < 40:
                    passthrough_map[i] = chunk
                else:
                    processable.append((i, chunk))
            
            if not processable:
                results.extend([passthrough_map.get(i, batch[i]) for i in range(len(batch))])
                continue
            
            try:
                # Prepare inputs with prefix if needed
                texts = []
                for _, chunk in processable:
                    if "flan-t5" in self.summarizer_model_name.lower():
                        texts.append(f"summarize: {chunk}")
                    else:
                        texts.append(chunk)
                
                # Determine length bounds from the first chunk (they'll be similar)
                sample_tokens = self._fast_token_estimate(processable[0][1])
                min_len, max_len = self._get_length_bounds(length_setting, sample_tokens, model_max_len)
                
                do_sample = num_beams == 1 and temperature > 0.1
                
                if self._is_onnx_model:
                    batch_inputs = self.tokenizer(texts, max_length=model_max_len, truncation=True,
                                                  padding=True, return_tensors="pt")
                else:
                    device = get_device_str()
                    batch_inputs = self.tokenizer(texts, max_length=model_max_len, truncation=True,
                                                  padding=True, return_tensors="pt").to(device)
                
                with torch.inference_mode():
                    batch_ids = self.summarizer.generate(
                        batch_inputs["input_ids"],
                        attention_mask=batch_inputs["attention_mask"],
                        max_length=max_len,
                        min_length=min_len,
                        do_sample=do_sample,
                        temperature=temperature if do_sample else None,
                        num_beams=num_beams,
                        length_penalty=length_penalty,
                        early_stopping=True if num_beams > 1 else False,
                        no_repeat_ngram_size=3
                    )
                
                decoded = [self.tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in batch_ids]
                
                # Reconstruct batch results with passthroughs
                batch_results = [None] * len(batch)
                for decoded_idx, (orig_idx, _) in enumerate(processable):
                    batch_results[orig_idx] = decoded[decoded_idx]
                for idx, text in passthrough_map.items():
                    batch_results[idx] = text
                results.extend(batch_results)
                
            except Exception as e:
                print(f"Batch inference failed ({e}), falling back to sequential...")
                for _, chunk in processable:
                    results.append(self.summarize_chunk(chunk, length_setting, temperature, num_beams, length_penalty))
                for idx in sorted(passthrough_map.keys()):
                    results.insert(batch_start + idx, passthrough_map[idx])
        
        return results

    def generate_summary(self, text: str, length_setting: str = "medium", 
                         temperature: float = 0.7, num_beams: int = 4, 
                         length_penalty: float = 2.0, progress_callback=None) -> Tuple[str, int]:
        """
        Generates a summary of the text. For large documents, automatically applies
        extractive TF-IDF pre-compression before chunked abstractive summarization.
        Uses batch inference for better CPU utilization.
        
        Performance characteristics (on CPU, with ONNX + 6-6 model):
        - <1000 tokens:  single-pass     (~1-2s)
        - 1000-5000:     chunked         (~4-8s)
        - >5000 tokens:  compress+chunked (~10-20s, even for 2.5MB files)
        
        Returns (summary_text, number_of_chunks_processed).
        """
        # Determine model_max_length dynamically
        model_max_len = getattr(self.tokenizer, "model_max_length", 1024)
        if not isinstance(model_max_len, int) or model_max_len > 4096 or model_max_len <= 0:
            model_max_len = 1024

        MAX_CHUNK_TOKENS = model_max_len
        MAX_CHUNKS = 20            # Hard cap — increased from 15 for more comprehensive coverage
        COMPRESS_THRESHOLD = MAX_CHUNK_TOKENS * 5  # Tokens above which extractive compression activates
        
        # Phase 1: Fast estimation to decide processing strategy
        estimated_tokens = self._fast_token_estimate(text)
        working_text = text
        
        # Phase 2: Extractive pre-compression for large documents
        if estimated_tokens > COMPRESS_THRESHOLD:
            if progress_callback:
                word_count = len(text.split())
                progress_callback(0.05, f"Large document (~{word_count:,} words). Running extractive compression...")
            
            # Target: enough sentences to fill ~MAX_CHUNKS chunks
            # Increased retention (250 sentences) for more comprehensive summaries
            target_sentences = min(250, max(100, MAX_CHUNKS * 12))
            working_text = self.extractive_compress(
                text, max_sentences=target_sentences, progress_callback=progress_callback
            )
            
            if progress_callback:
                compressed_words = len(working_text.split())
                progress_callback(0.15, f"Compressed to {compressed_words:,} words. Starting AI summarization...")
        
        # Phase 3: Check if single-pass summarization is possible
        total_tokens = self._fast_token_estimate(working_text)
        
        if total_tokens <= MAX_CHUNK_TOKENS:
            if progress_callback:
                progress_callback(0.3, "Processing document in a single pass...")
            summary = self.summarize_chunk(
                working_text, length_setting,
                temperature=temperature, num_beams=num_beams, length_penalty=length_penalty
            )
            if progress_callback:
                progress_callback(1.0, "Summarization complete!")
            return summary, 1
        
        # Phase 4: Hierarchical chunked summarization
        chunks = self.chunk_document(working_text, MAX_CHUNK_TOKENS)
        
        # Cap chunks with evenly-spaced sampling if still too many
        if len(chunks) > MAX_CHUNKS:
            if progress_callback:
                progress_callback(0.18, f"Sampling {MAX_CHUNKS} representative chunks from {len(chunks)}...")
            step = len(chunks) / MAX_CHUNKS
            chunks = [chunks[int(i * step)] for i in range(MAX_CHUNKS)]
        
        num_chunks = len(chunks)
        
        # Auto-reduce beam count on CPU (always 2 beams on CPU, full beams on GPU)
        effective_beams = num_beams
        _, dev_str = get_device()
        if dev_str == "cpu":
            effective_beams = min(num_beams, 2)
        elif num_chunks > 6:
            effective_beams = min(num_beams, 2)
        
        if progress_callback:
            progress_callback(0.2, f"Summarizing {num_chunks} chunks (beams={effective_beams})...")
        
        # Use batch inference for better throughput (local mode only)
        if not self.hf_api_token:
            chunk_summaries = []
            batch_size = 3
            for batch_start in range(0, num_chunks, batch_size):
                batch_end = min(batch_start + batch_size, num_chunks)
                batch_chunks = chunks[batch_start:batch_end]
                
                if progress_callback:
                    pct = 0.2 + 0.55 * (batch_end / num_chunks)
                    progress_callback(pct, f"Summarizing chunks {batch_start+1}-{batch_end} of {num_chunks} (batch)...")
                
                batch_results = self._summarize_batch(
                    batch_chunks, length_setting,
                    temperature=temperature, num_beams=effective_beams,
                    length_penalty=length_penalty, batch_size=batch_size
                )
                chunk_summaries.extend(batch_results)
        else:
            # API mode: sequential (API handles batching internally)
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                if progress_callback:
                    pct = 0.2 + 0.55 * ((i + 1) / num_chunks)
                    progress_callback(pct, f"Summarizing chunk {i+1} of {num_chunks}...")
                summary = self.summarize_chunk(
                    chunk, length_setting,
                    temperature=temperature, num_beams=effective_beams, length_penalty=length_penalty
                )
                chunk_summaries.append(summary)
        
        # Phase 5: Merge chunk summaries
        combined_text = " ".join(chunk_summaries)
        combined_tokens = self._fast_token_estimate(combined_text)
        
        # Recursive merge if combined summaries still exceed chunk limit
        # Increased from 3 to 4 levels for very long documents
        recursion_level = 1
        while combined_tokens > MAX_CHUNK_TOKENS and recursion_level < 4:
            if progress_callback:
                progress_callback(0.78 + 0.04 * recursion_level,
                                  f"Merging chunk summaries (Level {recursion_level})...")
            re_chunks = self.chunk_document(combined_text, MAX_CHUNK_TOKENS)
            re_summaries = self._summarize_batch(
                re_chunks, length_setting,
                temperature=temperature, num_beams=min(effective_beams, 2),
                length_penalty=length_penalty, batch_size=3
            )
            combined_text = " ".join(re_summaries)
            combined_tokens = self._fast_token_estimate(combined_text)
            recursion_level += 1
        
        # Phase 6: Final polish pass (use full beam count for quality)
        if length_setting == "long" and num_chunks > 1:
            if progress_callback:
                progress_callback(1.0, "Summarization complete!")
            clean_gpu_memory()
            return combined_text, num_chunks
            
        if progress_callback:
            progress_callback(0.9, "Generating final executive summary...")
        
        final_summary = self.summarize_chunk(
            combined_text, length_setting,
            temperature=temperature, num_beams=num_beams, length_penalty=length_penalty
        )
        
        if progress_callback:
            progress_callback(1.0, "Summarization complete!")
        
        clean_gpu_memory()
        return final_summary, num_chunks

    def classify_topics(self, text: str, candidate_topics: List[str]) -> Dict[str, float]:
        """
        Performs topic detection using Zero-Shot classification pipeline.
        Returns a dictionary mapping topic to confidence score.
        """
        # Crop text to fit zero-shot classifier window if needed (~500 words is safe)
        words = text.split()
        cropped_text = " ".join(words[:500])
        
        try:
            if self.hf_api_token:
                payload = {
                    "inputs": cropped_text,
                    "parameters": {"candidate_labels": candidate_topics, "multi_label": False}
                }
                result = self._query_hf_api(self.classifier_model_name, payload)
            else:
                result = self.classifier(cropped_text, candidate_labels=candidate_topics, multi_label=False)
            topic_scores = dict(zip(result['labels'], result['scores']))
            return topic_scores
        except Exception as e:
            print(f"Error in topic classification: {str(e)}")
            # Return uniform fallback distribution
            return {topic: 1.0 / len(candidate_topics) for topic in candidate_topics}

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Performs sentiment analysis using Zero-Shot classification.
        Labels: Positive, Negative, Neutral.
        Returns a dictionary mapping label to confidence score.
        """
        words = text.split()
        cropped_text = " ".join(words[:400])
        
        candidate_labels = ["Positive", "Negative", "Neutral"]
        try:
            if self.hf_api_token:
                payload = {
                    "inputs": cropped_text,
                    "parameters": {"candidate_labels": candidate_labels, "multi_label": False}
                }
                result = self._query_hf_api(self.classifier_model_name, payload)
            else:
                result = self.classifier(cropped_text, candidate_labels=candidate_labels, multi_label=False)
            sentiment_scores = dict(zip(result['labels'], result['scores']))
            return sentiment_scores
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            return {"Positive": 0.33, "Negative": 0.33, "Neutral": 0.34}

    def extract_keywords(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Extracts key phrases using KeyBERT.
        Returns a list of tuples: (keyword/phrase, score).
        """
        try:
            # Limit text input length to KeyBERT limit to prevent performance degradation
            words = text.split()
            cropped_text = " ".join(words[:1500])
            
            keywords = self.keybert.extract_keywords(
                cropped_text,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                use_maxsum=True,
                nr_candidates=20,
                top_n=top_n
            )
            return keywords
        except Exception as e:
            print(f"Error in keyword extraction: {str(e)}")
            # Fallback keyword extraction: frequency based
            from sklearn.feature_extraction.text import TfidfVectorizer
            try:
                vectorizer = TfidfVectorizer(max_features=top_n, stop_words='english')
                vectorizer.fit([text])
                features = vectorizer.get_feature_names_out()
                return [(word, 1.0 - (i * 0.05)) for i, word in enumerate(features)]
            except Exception:
                return []

    def format_summary_mode(self, original_text: str, summary_text: str, mode: str) -> str:
        """
        Post-processes the summary text into structured modes based on user choice:
        - Executive Summary (Standard Paragraphs)
        - Bullet Point Summary (Prefixes sentences with bullets)
        - Detailed Summary (Expanded paragraphs and highlights)
        - Meeting Notes Summary (Topics, Decisions, Action Items)
        - Research Paper Summary (Abstract, Methodology/Results, Conclusion)
        - Key Insights Summary (High-impact bulleted facts with icons)
        """
        sentences = split_text_into_sentences(summary_text)
        if not sentences:
            return ""
            
        if mode == "Bullet Point Summary":
            return "\n".join([f"• {sentence}" for sentence in sentences])
            
        elif mode == "Key Insights Summary":
            icons = ["💡", "🔑", "📌", "⚡", "🔍", "🎯", "📈", "⚙️"]
            insights = []
            for i, sentence in enumerate(sentences):
                icon = icons[i % len(icons)]
                insights.append(f"{icon} **Insight {i+1}:** {sentence}")
            return "\n\n".join(insights)
            
        elif mode == "Detailed Summary":
            # Group sentences into larger paragraph structures
            paragraphs = []
            current_paragraph = []
            for idx, sentence in enumerate(sentences):
                current_paragraph.append(sentence)
                if (idx + 1) % 3 == 0 or idx == len(sentences) - 1:
                    paragraphs.append(" ".join(current_paragraph))
                    current_paragraph = []
            return "\n\n".join(paragraphs)
            
        elif mode == "Meeting Notes Summary":
            # Classify sentences using Zero-Shot classification (batched for performance)
            labels = ["Discussion Topic/Context", "Key Decision/Agreement", "Action Item/Task Assignment"]
            groups = {label: [] for label in labels}
            
            try:
                if self.hf_api_token:
                    payload = {
                        "inputs": sentences,
                        "parameters": {"candidate_labels": labels, "multi_label": False}
                    }
                    results = self._query_hf_api(self.classifier_model_name, payload)
                else:
                    results = self.classifier(sentences, candidate_labels=labels, multi_label=False)
                
                if isinstance(results, dict):
                    results = [results]
                    
                for idx, sentence in enumerate(sentences):
                    best_label = results[idx]['labels'][0]
                    groups[best_label].append(sentence)
            except Exception as e:
                print(f"Error in meeting notes sentence batch classification: {e}")
                groups["Discussion Topic/Context"].extend(sentences)
            
            # Format output
            output = []
            output.append("### 🗓️ Meeting Notes Summary")
            
            output.append("\n#### 🗣️ Key Discussions & Context")
            if groups["Discussion Topic/Context"]:
                output.extend([f"- {s}" for s in groups["Discussion Topic/Context"]])
            else:
                output.append("_No general discussion points identified._")
                
            output.append("\n#### 🤝 Decisions Reached")
            if groups["Key Decision/Agreement"]:
                output.extend([f"- **Approved:** {s}" for s in groups["Key Decision/Agreement"]])
            else:
                output.append("_No explicit decisions identified._")
                
            output.append("\n#### 📋 Action Items & Deliverables")
            if groups["Action Item/Task Assignment"]:
                output.extend([f"- [ ] {s}" for s in groups["Action Item/Task Assignment"]])
            else:
                output.append("_No action items or assignments identified._")
                
            return "\n".join(output)
            
        elif mode == "Research Paper Summary":
            # Classify sentences into Research sections (batched for performance)
            labels = ["Objective/Hypothesis", "Methodology/Implementation", "Key Finding/Result", "Conclusion/Future Work"]
            groups = {label: [] for label in labels}
            
            try:
                if self.hf_api_token:
                    payload = {
                        "inputs": sentences,
                        "parameters": {"candidate_labels": labels, "multi_label": False}
                    }
                    results = self._query_hf_api(self.classifier_model_name, payload)
                else:
                    results = self.classifier(sentences, candidate_labels=labels, multi_label=False)
                
                if isinstance(results, dict):
                    results = [results]
                    
                for idx, sentence in enumerate(sentences):
                    best_label = results[idx]['labels'][0]
                    groups[best_label].append(sentence)
            except Exception as e:
                print(f"Error in research paper sentence batch classification: {e}")
                groups["Objective/Hypothesis"].extend(sentences)
                    
            output = []
            output.append("### 📑 Scientific/Research Paper Synthesis")
            
            output.append("\n#### 🎯 Objectives & Scope")
            if groups["Objective/Hypothesis"]:
                output.extend([f"- {s}" for s in groups["Objective/Hypothesis"]])
            else:
                output.append("_No specific hypothesis or objective identified._")
                
            output.append("\n#### 🛠️ Methodology & Experimental Setup")
            if groups["Methodology/Implementation"]:
                output.extend([f"- {s}" for s in groups["Methodology/Implementation"]])
            else:
                output.append("_No methodology details identified._")
                
            output.append("\n#### 📊 Key Findings & Results")
            if groups["Key Finding/Result"]:
                output.extend([f"- **Key Result:** {s}" for s in groups["Key Finding/Result"]])
            else:
                output.append("_No quantitative or key findings identified._")
                
            output.append("\n#### 🏁 Conclusions & Theoretical Implications")
            if groups["Conclusion/Future Work"]:
                output.extend([f"- {s}" for s in groups["Conclusion/Future Work"]])
            else:
                output.append("_No scientific conclusions identified._")
                
            return "\n".join(output)
            
        else:  # Executive Summary (Standard paragraphs)
            paragraphs = []
            current_paragraph = []
            for idx, sentence in enumerate(sentences):
                current_paragraph.append(sentence)
                if (idx + 1) % 4 == 0 or idx == len(sentences) - 1:
                    paragraphs.append(" ".join(current_paragraph))
                    current_paragraph = []
            return "\n\n".join(paragraphs)

    def generate_gemini_analysis(self, text: str, mode: str, length: str, candidate_topics: List[str]) -> Dict[str, Any]:
        """
        Uses Google Gemini 1.5 Flash API to perform document analysis.
        Returns a dictionary containing:
        - summary: raw executive summary
        - formatted_summary: formatted summary matching requested mode
        - keywords: list of (keyword, score) tuples
        - topics: dictionary of topic confidence scores
        - sentiment: dictionary of sentiment confidence scores
        """
        import requests
        import json
        
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_api_key}"
        headers = {"Content-Type": "application/json"}
        
        # Build prompt instructing Gemini to return structured JSON
        prompt = f"""You are a professional document analysis AI system.
Analyze the document text provided below and generate a structured summary, keywords, topic classifications, and sentiment metrics.
You MUST respond with a single, valid JSON object containing the following keys:
- "summary": A standard paragraphs-based executive summary.
  Guidelines for summary length:
  * If the length setting is "long", make the summary highly comprehensive, detailed, and complete, ensuring it covers all sections, details, and contexts of the document in depth (at least 600-1200 words for larger documents). Do not omit crucial context.
  * If the length setting is "medium", make the summary moderately detailed (300-500 words).
  * If the length setting is "short", make the summary concise and focused (100-200 words).
- "formatted_summary": A formatted version of the summary matching the requested mode "{mode}". It must adhere to the same length guidelines as above, but formatted according to the mode:
  * "Executive Summary": Balanced summary paragraphs.
  * "Bullet Point Summary": Prepend sentences with bullet points ("• ").
  * "Detailed Summary": Comprehensive paragraphs detailing all main points.
  * "Meeting Notes Summary": Structured with headers "### 🗓️ Meeting Notes Summary", "#### 🗣️ Key Discussions", "#### 🤝 Decisions Reached", and "#### 📋 Action Items".
  * "Research Paper Summary": Structured with headers "### 📑 Scientific/Research Paper Synthesis", "#### 🎯 Objectives", "#### 🛠️ Methodology", "#### 📊 Key Findings", and "#### 🏁 Conclusions".
  * "Key Insights Summary": Highlighted high-impact points prepended with emojis (💡, 🔑, 📌, etc.).
- "keywords": An array of at most 10 key phrases and terms with relevance scores, formatted as [["phrase", score], ...]. Example: [["machine learning", 0.95], ["model accuracy", 0.85]].
- "topics": A dictionary mapping each of the candidate topics to a confidence score between 0.0 and 1.0. The scores must sum to approximately 1.0. Candidate topics: {candidate_topics}.
- "sentiment": A dictionary mapping "Positive", "Negative", and "Neutral" to a confidence score between 0.0 and 1.0. The scores must sum to approximately 1.0.

Requested Summary Length setting: "{length}".

Document Text:
{text}
"""

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json"
            }
        }
        
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=40)
            if response.status_code == 200:
                res_json = response.json()
                text_response = res_json['candidates'][0]['content']['parts'][0]['text'].strip()
                # Clean potential markdown wrapping
                if text_response.startswith("```"):
                    text_response = re.sub(r"^```(?:json)?\n", "", text_response)
                    text_response = re.sub(r"\n```$", "", text_response)
                    text_response = text_response.strip()
                data = json.loads(text_response)
                return data
            else:
                raise ValueError(f"Gemini API returned status code {response.status_code}: {response.text}")
        except Exception as e:
            raise ValueError(f"Error querying Gemini API: {str(e)}")
