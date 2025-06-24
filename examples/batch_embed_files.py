#!/usr/bin/env python3
"""
Batch embed files using Ollama and store in VectorClusterStore.

This script reads file paths from stdin, processes them in batches,
embeds their content using Ollama, and stores the embeddings in a
vectl device/file.

Usage:
    find . -name "*.txt" | python batch_embed_files.py [options]
    ls *.md | python batch_embed_files.py --batch-size 10
    echo "/path/to/file.txt" | python batch_embed_files.py
"""

import sys
import os
import json
import time
import argparse
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import requests
from pathlib import Path
import mimetypes
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add build directory to path for vector store imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(os.path.join(parent_dir, "build"))
import vector_cluster_store_py

# Default configuration
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/embed"
DEFAULT_MODEL = "nomic-embed-text:latest"
DEFAULT_VECTOR_DIM = 768
DEFAULT_BATCH_SIZE = 5
DEFAULT_DEVICE_PATH = "./vector_store.bin"
DEFAULT_LOG_FILE = "./batch_embed.log"
DEFAULT_METADATA_FILE = "./batch_embed_metadata.json"
DEFAULT_MAX_FILE_SIZE = 1024 * 1024  # 1MB
DEFAULT_MAX_CONTENT_LENGTH = 8192  # Characters to process per file

# Thread-safe counter for progress tracking
class ProgressCounter:
    def __init__(self):
        self.lock = threading.Lock()
        self.processed = 0
        self.failed = 0
        self.total = 0
    
    def increment_processed(self):
        with self.lock:
            self.processed += 1
    
    def increment_failed(self):
        with self.lock:
            self.failed += 1
    
    def set_total(self, total):
        with self.lock:
            self.total = total
    
    def get_stats(self):
        with self.lock:
            return self.processed, self.failed, self.total


def read_file_content(file_path: str, max_size: int, max_length: int) -> Optional[str]:
    """Read and preprocess file content for embedding."""
    try:
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            print(f"Warning: File not found: {file_path}")
            return None
        
        # Check file size
        if path.stat().st_size > max_size:
            print(f"Warning: File too large (>{max_size} bytes): {file_path}")
            return None
        
        # Determine file type
        mime_type, _ = mimetypes.guess_type(file_path)
        
        # Read file content based on type
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"Warning: Could not read file {file_path}: {e}")
            return None
        
        # Truncate if necessary
        if len(content) > max_length:
            content = content[:max_length] + "..."
            print(f"Info: Truncated content for {file_path} to {max_length} characters")
        
        # Basic preprocessing
        content = content.strip()
        if not content:
            print(f"Warning: Empty file: {file_path}")
            return None
        
        return content
        
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def batch_embed_ollama(texts: List[str], model: str, api_url: str) -> List[List[float]]:
    """Send batch embedding request to Ollama."""
    try:
        payload = {
            "model": model,
            "input": texts  # Ollama supports batch input
        }
        
        response = requests.post(api_url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        embeddings = data.get("embeddings", [])
        
        # Validate embeddings
        if len(embeddings) != len(texts):
            print(f"Warning: Expected {len(texts)} embeddings, got {len(embeddings)}")
            return []
        
        return embeddings
        
    except requests.exceptions.Timeout:
        print(f"Error: Ollama request timed out")
        return []
    except Exception as e:
        print(f"Error calling Ollama API: {e}")
        return []


def process_batch(
    file_batch: List[Tuple[str, str]], 
    store: vector_cluster_store_py.VectorClusterStore,
    metadata_dict: Dict,
    model: str,
    api_url: str,
    vector_dim: int,
    progress: ProgressCounter
) -> int:
    """Process a batch of files: embed and store."""
    if not file_batch:
        return 0
    
    # Extract texts from batch
    texts = [content for _, content in file_batch]
    file_paths = [path for path, _ in file_batch]
    
    print(f"\nProcessing batch of {len(texts)} files...")
    start_time = time.time()
    
    # Get embeddings from Ollama
    embeddings = batch_embed_ollama(texts, model, api_url)
    
    if not embeddings:
        print(f"Failed to get embeddings for batch")
        for _ in file_batch:
            progress.increment_failed()
        return 0
    
    embed_time = time.time() - start_time
    print(f"Batch embedding completed in {embed_time:.2f}s")
    
    # Store each embedding
    stored_count = 0
    for i, (file_path, content) in enumerate(file_batch):
        if i >= len(embeddings):
            print(f"Warning: No embedding for file {file_path}")
            progress.increment_failed()
            continue
        
        embedding = embeddings[i]
        
        # Validate embedding dimension
        if len(embedding) != vector_dim:
            if len(embedding) > vector_dim:
                embedding = embedding[:vector_dim]
            else:
                embedding = embedding + [0.0] * (vector_dim - len(embedding))
        
        # Prepare metadata
        metadata_entry = {
            "file_path": file_path,
            "content_preview": content[:200] + "..." if len(content) > 200 else content,
            "content_length": len(content),
            "timestamp": datetime.now().isoformat(),
            "embedding_model": model,
            "file_size": Path(file_path).stat().st_size if Path(file_path).exists() else 0
        }
        metadata_json = json.dumps(metadata_entry)
        
        # Store vector
        try:
            vector_id = metadata_dict["next_id"]
            success = store.store_vector(vector_id, embedding, metadata_json)
            
            if success:
                metadata_dict["entries"][str(vector_id)] = metadata_entry
                metadata_dict["next_id"] = vector_id + 1
                stored_count += 1
                progress.increment_processed()
                
                # Print progress
                processed, failed, total = progress.get_stats()
                print(f"[{processed}/{total}] Stored: {file_path} (ID: {vector_id})")
            else:
                print(f"Failed to store embedding for: {file_path}")
                progress.increment_failed()
                
        except Exception as e:
            print(f"Error storing embedding for {file_path}: {e}")
            progress.increment_failed()
    
    return stored_count


def save_metadata(metadata: Dict, metadata_file: str):
    """Save metadata to file."""
    try:
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"Error saving metadata: {e}")


def load_metadata(metadata_file: str, vector_dim: int) -> Dict:
    """Load metadata from file."""
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading metadata: {e}")
    
    return {
        "next_id": 0,
        "entries": {},
        "vector_dim": vector_dim,
        "created_at": datetime.now().isoformat()
    }


def main():
    parser = argparse.ArgumentParser(
        description="Batch embed files using Ollama and store in VectorClusterStore",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "device_path",
        nargs="?",
        default=DEFAULT_DEVICE_PATH,
        help=f"Path to vector store device/file (default: {DEFAULT_DEVICE_PATH})"
    )
    
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama API URL (default: {DEFAULT_OLLAMA_URL})"
    )
    
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Embedding model (default: {DEFAULT_MODEL})"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of files to process in each batch (default: {DEFAULT_BATCH_SIZE})"
    )
    
    parser.add_argument(
        "--max-file-size",
        type=int,
        default=DEFAULT_MAX_FILE_SIZE,
        help=f"Maximum file size in bytes (default: {DEFAULT_MAX_FILE_SIZE})"
    )
    
    parser.add_argument(
        "--max-content-length",
        type=int,
        default=DEFAULT_MAX_CONTENT_LENGTH,
        help=f"Maximum content length per file (default: {DEFAULT_MAX_CONTENT_LENGTH})"
    )
    
    parser.add_argument(
        "--vector-dim",
        type=int,
        default=DEFAULT_VECTOR_DIM,
        help=f"Vector dimension (default: {DEFAULT_VECTOR_DIM})"
    )
    
    parser.add_argument(
        "--log-file",
        default=None,
        help=f"Log file path (default: auto-derived from device path)"
    )
    
    parser.add_argument(
        "--metadata-file",
        default=None,
        help=f"Metadata file path (default: auto-derived from device path)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually embedding"
    )
    
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing even if some files fail"
    )
    
    args = parser.parse_args()
    
    # Auto-derive log and metadata paths from device path if not specified
    device_dir = os.path.dirname(args.device_path) if os.path.dirname(args.device_path) else "."
    device_name = os.path.splitext(os.path.basename(args.device_path))[0]
    
    # Ensure directory exists
    os.makedirs(device_dir, exist_ok=True)
    
    if args.log_file is None:
        args.log_file = os.path.join(device_dir, f"{device_name}.log")
    
    if args.metadata_file is None:
        args.metadata_file = os.path.join(device_dir, f"{device_name}_metadata.json")
    
    # Check if using block device
    if args.device_path.startswith("/dev/"):
        print(f"⚠️  WARNING: Using block device: {args.device_path}")
        print("This requires elevated permissions and can cause data loss.")
        if not args.dry_run:
            confirm = input("Continue? (y/n): ")
            if confirm.lower() != 'y':
                print("Aborted.")
                return
    
    print(f"Batch Embed Files Configuration:")
    print(f"  Vector store: {args.device_path}")
    print(f"  Model: {args.model}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max file size: {args.max_file_size} bytes")
    print(f"  Max content length: {args.max_content_length} chars")
    print(f"  Vector dimension: {args.vector_dim}")
    print()
    
    # Read file paths from stdin
    print("Reading file paths from stdin...")
    file_paths = []
    for line in sys.stdin:
        path = line.strip()
        if path:
            file_paths.append(path)
    
    if not file_paths:
        print("No file paths provided via stdin")
        return
    
    print(f"Found {len(file_paths)} file paths")
    
    if args.dry_run:
        print("\n[DRY RUN] Would process the following files:")
        for path in file_paths[:10]:
            print(f"  - {path}")
        if len(file_paths) > 10:
            print(f"  ... and {len(file_paths) - 10} more")
        return
    
    # Initialize vector store
    try:
        logger = vector_cluster_store_py.Logger(args.log_file)
        store = vector_cluster_store_py.VectorClusterStore(logger)
        
        if not store.initialize(args.device_path, "kmeans", args.vector_dim, 10):
            print(f"Error initializing vector store. Check log: {args.log_file}")
            return
            
        print("Vector store initialized successfully")
    except Exception as e:
        print(f"Failed to initialize vector store: {e}")
        return
    
    # Load metadata
    metadata = load_metadata(args.metadata_file, args.vector_dim)
    initial_count = len(metadata["entries"])
    print(f"Loaded metadata with {initial_count} existing entries")
    
    # Progress tracking
    progress = ProgressCounter()
    progress.set_total(len(file_paths))
    
    # Process files in batches
    batch = []
    total_stored = 0
    start_time = time.time()
    
    for file_path in file_paths:
        # Read file content
        content = read_file_content(
            file_path, 
            args.max_file_size, 
            args.max_content_length
        )
        
        if content:
            batch.append((file_path, content))
        else:
            progress.increment_failed()
            if not args.continue_on_error:
                print(f"Aborting due to error with file: {file_path}")
                break
        
        # Process batch when full
        if len(batch) >= args.batch_size:
            stored = process_batch(
                batch, store, metadata, args.model, 
                args.ollama_url, args.vector_dim, progress
            )
            total_stored += stored
            
            # Save metadata periodically
            save_metadata(metadata, args.metadata_file)
            batch = []
    
    # Process remaining files
    if batch:
        stored = process_batch(
            batch, store, metadata, args.model,
            args.ollama_url, args.vector_dim, progress
        )
        total_stored += stored
    
    # Final metadata save
    save_metadata(metadata, args.metadata_file)
    
    # Summary
    elapsed_time = time.time() - start_time
    processed, failed, total = progress.get_stats()
    
    print("\n" + "=" * 60)
    print("BATCH EMBEDDING COMPLETE")
    print("=" * 60)
    print(f"Total files: {total}")
    print(f"Successfully processed: {processed}")
    print(f"Failed: {failed}")
    print(f"Total time: {elapsed_time:.2f}s")
    print(f"Average time per file: {elapsed_time/total:.2f}s" if total > 0 else "N/A")
    print(f"Vectors stored: {total_stored}")
    print(f"Total vectors in store: {len(metadata['entries'])}")
    print(f"Metadata saved to: {args.metadata_file}")


if __name__ == "__main__":
    main()