#!/usr/bin/env python3
"""
Command-line vector search utility for VectorClusterStore.

Usage:
    python vector_search.py "search query"
    echo "search query" | python vector_search.py
    
Options:
    --store PATH        Path to vector store (default: ../vector_store.bin)
    --metadata PATH     Path to metadata JSON (default: ../batch_embed_metadata.json)
    --model NAME        Ollama model name (default: nomic-embed-text:latest)
    --top-k N          Number of results to return (default: 10)
    --threshold FLOAT  Similarity threshold (0-1, default: 0.0)
    --output FORMAT    Output format: json, text (default: json)
    --no-embed         Use raw query as vector (for testing)
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
import ollama

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "build"))
import vector_cluster_store_py as vcs


def load_metadata(metadata_path):
    """Load metadata from JSON file."""
    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"embeddings": {}}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {metadata_path}", file=sys.stderr)
        return {"embeddings": {}}


def get_embedding(text, model_name):
    """Get embedding vector from Ollama."""
    try:
        response = ollama.embeddings(model=model_name, prompt=text)
        return np.array(response['embedding'], dtype=np.float32)
    except Exception as e:
        print(f"Error getting embedding: {e}", file=sys.stderr)
        return None


def search_vectors(store, query_vector, top_k, threshold):
    """Search for similar vectors in the store."""
    results = store.find_similar_vectors(query_vector, top_k)
    
    # Filter by threshold if specified
    if threshold > 0:
        results = [(idx, score) for idx, score in results if score >= threshold]
    
    return results


def format_results(results, metadata, output_format):
    """Format search results based on output format."""
    embeddings_metadata = metadata.get("entries", metadata.get("embeddings", {}))
    
    if output_format == "json":
        output = {
            "results": []
        }
        
        for idx, score in results:
            result = {
                "index": idx,
                "score": float(score)
            }
            
            # Add metadata if available
            str_idx = str(idx)
            if str_idx in embeddings_metadata:
                result["metadata"] = embeddings_metadata[str_idx]
            
            output["results"].append(result)
        
        return json.dumps(output, indent=2)
    
    else:  # text format
        lines = []
        for i, (idx, score) in enumerate(results):
            lines.append(f"\n{i+1}. Index: {idx}, Score: {score:.4f}")
            
            # Add metadata if available
            str_idx = str(idx)
            if str_idx in embeddings_metadata:
                meta = embeddings_metadata[str_idx]
                if "file_path" in meta:
                    lines.append(f"   File: {meta['file_path']}")
                if "chunk_index" in meta:
                    lines.append(f"   Chunk: {meta['chunk_index']}")
                if "content_preview" in meta:
                    lines.append(f"   Content: {meta['content_preview']}")
                elif "content" in meta:
                    content_preview = meta['content'][:200] + "..." if len(meta['content']) > 200 else meta['content']
                    lines.append(f"   Content: {content_preview}")
        
        return "\n".join(lines) if lines else "No results found."


def main():
    parser = argparse.ArgumentParser(description="Search vectors in VectorClusterStore")
    parser.add_argument("query", nargs="?", help="Search query (can also be piped via stdin)")
    parser.add_argument("--store", default="../vector_store.bin", help="Path to vector store")
    parser.add_argument("--metadata", default="../batch_embed_metadata.json", help="Path to metadata JSON")
    parser.add_argument("--model", default="nomic-embed-text:latest", help="Ollama model name")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to return")
    parser.add_argument("--threshold", type=float, default=0.0, help="Similarity threshold (0-1)")
    parser.add_argument("--output", choices=["json", "text"], default="json", help="Output format")
    parser.add_argument("--no-embed", action="store_true", help="Use raw query as vector (testing)")
    
    args = parser.parse_args()
    
    # Get query from argument or stdin
    if args.query:
        query = args.query
    elif not sys.stdin.isatty():
        query = sys.stdin.read().strip()
    else:
        print("Error: No query provided. Use as argument or pipe via stdin.", file=sys.stderr)
        sys.exit(1)
    
    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    store_path = script_dir / args.store
    metadata_path = script_dir / args.metadata
    
    # Check if store exists
    if not store_path.exists():
        print(f"Error: Vector store not found at {store_path}", file=sys.stderr)
        sys.exit(1)
    
    # Load metadata
    metadata = load_metadata(metadata_path)
    
    # Initialize store
    try:
        logger = vcs.Logger("/dev/null")  # Disable logging for CLI tool
        store = vcs.VectorClusterStore(logger)
        
        # Determine vector dimension from metadata or use default
        vector_dim = 768  # Default for nomic-embed-text
        entries = metadata.get("entries", metadata.get("embeddings", {}))
        if entries:
            # Get dimension from first entry if available
            first_key = next(iter(entries), None)
            if first_key and "vector_dim" in entries[first_key]:
                vector_dim = entries[first_key]["vector_dim"]
        elif "vector_dim" in metadata:
            vector_dim = metadata["vector_dim"]
        
        if not store.initialize(str(store_path), "kmeans", vector_dim, 10):
            print(f"Error: Failed to initialize vector store at {store_path}", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Error opening vector store: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Get query vector
    if args.no_embed:
        # For testing: parse query as comma-separated floats
        try:
            query_vector = np.array([float(x) for x in query.split(",")], dtype=np.float32)
        except ValueError:
            print("Error: --no-embed requires comma-separated float values", file=sys.stderr)
            sys.exit(1)
    else:
        # Get embedding from Ollama
        query_vector = get_embedding(query, args.model)
        if query_vector is None:
            sys.exit(1)
    
    # Perform search
    try:
        results = search_vectors(store, query_vector, args.top_k, args.threshold)
    except Exception as e:
        print(f"Error during search: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Format and output results
    output = format_results(results, metadata, args.output)
    print(output)


if __name__ == "__main__":
    main()