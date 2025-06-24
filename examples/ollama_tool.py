#!/usr/bin/env python3

import requests
import json
import time
import sys
import os
import argparse
import numpy as np
from datetime import datetime
from tabulate import tabulate

# Import our vector store Python bindings
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(os.path.join(parent_dir, "build"))
import vector_cluster_store_py


class OllamaVectorTool:
    def __init__(self, args):
        self.args = args
        self.ollama_api_url = f"{args.ollama_url}/api/embed"
        self.embedding_model = args.model
        self.vector_dim = args.dimension
        self.device_path = args.index
        
        # Auto-derive log and metadata paths from device path
        device_dir = os.path.dirname(self.device_path) if os.path.dirname(self.device_path) else "."
        device_name = os.path.splitext(os.path.basename(self.device_path))[0]
        
        # Ensure directory exists
        os.makedirs(device_dir, exist_ok=True)
        
        self.log_file = os.path.join(device_dir, f"{device_name}.log")
        self.metadata_file = os.path.join(device_dir, f"{device_name}_metadata.json")
        
        self.store = None
        self.metadata = None

    def init_vector_store(self):
        """Initialize vector store with proper error handling"""
        try:
            # Create the directory if it doesn't exist
            device_dir = os.path.dirname(self.device_path)
            if device_dir and not os.path.exists(device_dir):
                os.makedirs(device_dir, exist_ok=True)
            
            # Create logger
            logger = vector_cluster_store_py.Logger(self.log_file)

            # Create and initialize vector store
            self.store = vector_cluster_store_py.VectorClusterStore(logger)

            # Initialize with kmeans clustering
            if not self.store.initialize(self.device_path, "kmeans", self.vector_dim, self.args.clusters):
                print(f"Error initializing vector store. Please check the log file: {self.log_file}")
                sys.exit(1)

            if self.args.verbose:
                print("Vector store initialized successfully")
                print(f"Device path: {self.device_path}")
                print(f"Log file: {self.log_file}")
                print(f"Metadata file: {self.metadata_file}")

        except Exception as e:
            print(f"Error initializing vector store: {e}")
            if self.args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)

    def load_metadata(self):
        """Load metadata from file"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, "r") as f:
                    self.metadata = json.load(f)
            except Exception as e:
                if self.args.verbose:
                    print(f"Error loading metadata: {e}")
                self.metadata = {"next_id": 0, "entries": {}, "vector_dim": self.vector_dim}
        else:
            self.metadata = {"next_id": 0, "entries": {}, "vector_dim": self.vector_dim}

    def save_metadata(self):
        """Save metadata to file"""
        try:
            # Ensure metadata includes vector dimension
            self.metadata["vector_dim"] = self.vector_dim
            with open(self.metadata_file, "w") as f:
                json.dump(self.metadata, f)
        except Exception as e:
            if self.args.verbose:
                print(f"Error saving metadata: {e}")

    def get_embedding(self, text):
        """Get embedding from Ollama"""
        try:
            start_time = time.time()

            payload = {"model": self.embedding_model, "input": text}

            response = requests.post(self.ollama_api_url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()

            embedding = data["embeddings"][0]
            total_duration = data.get("total_duration", 0) / 1_000_000  # Convert to ms

            elapsed = time.time() - start_time

            if self.args.verbose:
                print(f"Embedding generated in {elapsed:.2f}s (Ollama processing: {total_duration:.2f}ms)")

            # Verify the embedding dimension
            if len(embedding) != self.vector_dim:
                if self.args.verbose:
                    print(f"Warning: Embedding dimension mismatch. Got {len(embedding)}, expected {self.vector_dim}")
                if len(embedding) > self.vector_dim:
                    return embedding[:self.vector_dim]
                else:
                    padded = embedding + [0.0] * (self.vector_dim - len(embedding))
                    return padded

            return embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            if self.args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)

    def embed_mode(self):
        """Handle embed mode - store vectors from input text"""
        self.init_vector_store()
        self.load_metadata()
        
        if self.args.verbose:
            print(f"Embedding mode - using model: {self.embedding_model}")
            print(f"Next ID: {self.metadata['next_id']}")
        
        # Handle input from stdin or arguments
        if self.args.text:
            texts = [self.args.text]
        else:
            if self.args.verbose:
                print("Reading from stdin (one text per line, Ctrl+D to finish)...")
            texts = [line.strip() for line in sys.stdin if line.strip()]
        
        success_count = 0
        for text in texts:
            embedding = self.get_embedding(text)
            
            # Prepare metadata string (JSON)
            metadata_entry = {
                "text": text,
                "timestamp": datetime.now().isoformat(),
                "embedding_model": self.embedding_model,
            }
            metadata_json = json.dumps(metadata_entry)
            
            try:
                vector_id = self.metadata["next_id"]
                success = self.store.store_vector(vector_id, embedding, metadata_json)
                
                if success:
                    if self.args.verbose:
                        print(f"Stored: ID {vector_id} - \"{text[:50]}...\"" if len(text) > 50 else f"Stored: ID {vector_id} - \"{text}\"")
                    
                    # Save text with ID for future reference
                    self.metadata["entries"][str(vector_id)] = metadata_entry
                    self.metadata["next_id"] = vector_id + 1
                    success_count += 1
                else:
                    print(f"Failed to store: \"{text}\"")
            except Exception as e:
                print(f"Error storing text: {e}")
                if self.args.verbose:
                    import traceback
                    traceback.print_exc()
        
        self.save_metadata()
        
        if self.args.format == "json":
            result = {
                "success": True,
                "stored_count": success_count,
                "next_id": self.metadata["next_id"]
            }
            print(json.dumps(result, indent=2))
        else:
            print(f"\nSuccessfully stored {success_count} embeddings")

    def search_mode(self):
        """Handle search mode - find similar vectors"""
        self.init_vector_store()
        self.load_metadata()
        
        if not self.args.query:
            print("Error: --query is required for search mode")
            sys.exit(1)
        
        if self.args.verbose:
            print(f"Search mode - using model: {self.embedding_model}")
            print(f"Query: \"{self.args.query}\"")
            print(f"Returning top {self.args.top_k} results")
        
        embedding = self.get_embedding(self.args.query)
        
        try:
            start_time = time.time()
            results = self.store.find_similar_vectors(embedding, self.args.top_k)
            elapsed = time.time() - start_time
            
            if self.args.verbose:
                print(f"Search completed in {elapsed*1000:.2f}ms")
            
            if results:
                # Prepare results with metadata
                formatted_results = []
                for id, similarity in sorted(results, key=lambda x: x[1], reverse=True):
                    entry = self.metadata["entries"].get(str(id), {})
                    formatted_results.append({
                        "id": id,
                        "score": round(similarity, 4),
                        "text": entry.get("text", "Unknown"),
                        "timestamp": entry.get("timestamp", "Unknown"),
                        "model": entry.get("embedding_model", "Unknown")
                    })
                
                if self.args.format == "json":
                    output = {
                        "query": self.args.query,
                        "results": formatted_results,
                        "search_time_ms": round(elapsed * 1000, 2)
                    }
                    print(json.dumps(output, indent=2))
                else:
                    # Table format
                    table_data = []
                    for r in formatted_results:
                        text_preview = r["text"][:60] + "..." if len(r["text"]) > 60 else r["text"]
                        table_data.append([r["id"], r["score"], text_preview, r["timestamp"]])
                    
                    print(f"\nTop {len(results)} matches for: \"{self.args.query}\"")
                    print(tabulate(table_data, headers=["ID", "Score", "Text", "Timestamp"], tablefmt="grid"))
            else:
                if self.args.format == "json":
                    print(json.dumps({"query": self.args.query, "results": [], "message": "No results found"}))
                else:
                    print("No results found")
                    
        except Exception as e:
            print(f"Error during search: {e}")
            if self.args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)

    def maintenance_mode(self):
        """Handle maintenance mode"""
        self.init_vector_store()
        
        if self.args.verbose:
            print("Performing maintenance...")
        
        try:
            success = self.store.perform_maintenance()
            
            if self.args.format == "json":
                print(json.dumps({"success": success, "message": "Maintenance completed" if success else "Maintenance failed"}))
            else:
                if success:
                    print("Maintenance completed successfully")
                else:
                    print("Maintenance failed")
        except Exception as e:
            print(f"Error during maintenance: {e}")
            if self.args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)

    def info_mode(self):
        """Print store information"""
        self.init_vector_store()
        self.load_metadata()
        
        print(f"\nVector Store Information:")
        print(f"========================")
        print(f"Index path: {self.device_path}")
        print(f"Metadata file: {self.metadata_file}")
        print(f"Log file: {self.log_file}")
        print(f"Vector dimension: {self.vector_dim}")
        print(f"Number of clusters: {self.args.clusters}")
        print(f"Embedding model: {self.embedding_model}")
        print(f"Next ID: {self.metadata['next_id']}")
        print(f"Stored entries: {len(self.metadata['entries'])}")
        
        print("\nStore Details:")
        self.store.print_store_info()

    def run(self):
        """Run the tool in the specified mode"""
        if self.args.mode == "embed":
            self.embed_mode()
        elif self.args.mode == "search":
            self.search_mode()
        elif self.args.mode == "maintenance":
            self.maintenance_mode()
        elif self.args.mode == "info":
            self.info_mode()
        else:
            print(f"Unknown mode: {self.args.mode}")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Ollama Vector Store Tool - Embed and search text using Ollama embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Embed a single text
  %(prog)s embed --text "Hello world"
  
  # Embed multiple texts from stdin
  echo -e "First text\\nSecond text" | %(prog)s embed
  
  # Search for similar texts
  %(prog)s search --query "Hello" --top-k 5
  
  # Search with JSON output
  %(prog)s search --query "Hello" --format json
  
  # Use custom index location
  %(prog)s embed --index /path/to/index.bin --text "Custom location"
  
  # Run maintenance
  %(prog)s maintenance
  
  # Show store information
  %(prog)s info
"""
    )
    
    # Mode selection
    parser.add_argument(
        "mode",
        choices=["embed", "search", "maintenance", "info"],
        help="Operation mode: embed text, search for similar, perform maintenance, or show info"
    )
    
    # Common arguments
    parser.add_argument(
        "--index",
        default="./vector_store.bin",
        help="Path to vector store index file (default: ./vector_store.bin)"
    )
    
    parser.add_argument(
        "--ollama-url",
        default="http://127.0.0.1:11434",
        help="Ollama API URL (default: http://127.0.0.1:11434)"
    )
    
    parser.add_argument(
        "--model",
        default="nomic-embed-text:latest",
        help="Ollama embedding model (default: nomic-embed-text:latest)"
    )
    
    parser.add_argument(
        "--dimension",
        type=int,
        default=768,
        help="Vector dimension (default: 768 for nomic-embed-text)"
    )
    
    parser.add_argument(
        "--clusters",
        type=int,
        default=10,
        help="Number of clusters for k-means (default: 10)"
    )
    
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    # Embed mode arguments
    parser.add_argument(
        "--text",
        help="Text to embed (for embed mode). If not provided, reads from stdin"
    )
    
    # Search mode arguments
    parser.add_argument(
        "--query",
        help="Query text for search mode"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return for search (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Create and run tool
    tool = OllamaVectorTool(args)
    tool.run()


if __name__ == "__main__":
    main()