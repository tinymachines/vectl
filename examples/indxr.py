#!/usr/bin/env python3
import requests
import json
import time
import sys
import os
import numpy as np
from datetime import datetime
import re

# Import our vector store Python bindings
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "build"))
import vector_cluster_store_py

# Configuration
OLLAMA_API_URL = "https://ollama.meatball.ai/api/embed"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_DIM = 768  # Dimension of nomic-embed-text embeddings
# DEVICE_PATH = "./vector_store.bin"  # Use current directory
DEVICE_PATH = "/dev/sdb"  # Use current directory
FILE_SIZE_MB = 128
METADATA_FILE = "./vector_store_metadata.json"  # Use current directory
LOG_FILE = "./vector_store.log"  # Log file for the logger

from enhanced_json_repair import EnhancedJSONRepair
# Create output directory
os.makedirs("repaired_samples", exist_ok=True)

# Initialize vector store with proper error handling
def init_vector_store():
    try:
        # Create logger
        logger = vector_cluster_store_py.Logger(LOG_FILE)

        # Create and initialize vector store
        store = vector_cluster_store_py.VectorClusterStore(logger)

        # Initialize with kmeans clustering
        if not store.initialize(DEVICE_PATH, "kmeans", VECTOR_DIM, 10):
            print(
                f"Error initializing vector store. Please check the log file: {LOG_FILE}"
            )
            sys.exit(1)

        print("Vector store initialized successfully")
        return store

    except Exception as e:
        print(f"Error initializing vector store: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


# Load and save metadata
def load_metadata():
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading metadata: {e}")
    return {"next_id": 0, "entries": {}, "vector_dim": VECTOR_DIM}


store = init_vector_store()
metadata = load_metadata()

def save_metadata(metadata):
    try:
        # Ensure metadata includes vector dimension
        metadata["vector_dim"] = VECTOR_DIM
        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f)
    except Exception as e:
        print(f"Error saving metadata: {e}")


# Get embedding from Ollama with proper error handling
def get_embedding(text):
    try:
        start_time = time.time()

        payload = {"model": EMBEDDING_MODEL, "input": text}

        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        data = response.json()

        embedding = data["embeddings"][0]
        total_duration = data.get("total_duration", 0) / 1_000_000  # Convert to ms

        elapsed = time.time() - start_time

        print(
            f"Embedding generated in {elapsed:.2f}s (Ollama processing: {total_duration:.2f}ms)"
        )
        print(f"SIZE {len(data['embeddings'][0])} ?= {VECTOR_DIM}")

        # Verify the embedding dimension
        if len(embedding) != VECTOR_DIM:
            print(
                f"Warning: Embedding dimension mismatch. Got {len(embedding)}, expected {VECTOR_DIM}"
            )
            if len(embedding) > VECTOR_DIM:
                print(f"Truncating to {VECTOR_DIM} dimensions")
                return embedding[:VECTOR_DIM]
            else:
                print(f"Padding to {VECTOR_DIM} dimensions")
                padded = embedding + [0.0] * (VECTOR_DIM - len(embedding))
                return padded

        return embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        # Return a dummy embedding for testing
        print("Returning dummy embedding for testing")
        return [0.1] * VECTOR_DIM


# Store text embedding
def store_text(store, text, metadata_dict):
    embedding = get_embedding(text)

    # Ensure embedding is not None and has the right dimension
    if embedding is None or len(embedding) != VECTOR_DIM:
        print(f"Invalid embedding, cannot store.")
        return metadata_dict["next_id"]

    # Prepare metadata string (JSON)
    metadata_entry = {
        "text": text,
        "timestamp": datetime.now().isoformat(),
        "embedding_model": EMBEDDING_MODEL,
    }
    metadata_json = json.dumps(metadata_entry)

    start_time = time.time()
    try:
        vector_id = metadata_dict["next_id"]
        success = store.store_vector(vector_id, embedding, metadata_json)
        elapsed = time.time() - start_time

        if success:
            print(f"Stored embedding with ID {vector_id} in {elapsed*1000:.2f}ms")
            # Save text with ID for future reference
            metadata_dict["entries"][str(vector_id)] = metadata_entry
            next_id = vector_id + 1
            metadata_dict["next_id"] = next_id
            save_metadata(metadata_dict)
            return next_id
        else:
            print(f"Failed to store embedding")
            return metadata_dict["next_id"]
    except Exception as e:
        print(f"Error in store_text: {e}")
        import traceback

        traceback.print_exc()
        return metadata_dict["next_id"]


# Retrieve vector by ID
def retrieve_vector(store, id, metadata):
    try:
        start_time = time.time()
        vector = store.retrieve_vector(id)
        elapsed = time.time() - start_time

        if len(vector) > 0:
            print(f"Retrieved vector {id} in {elapsed*1000:.2f}ms")
            print(f"Dimensions: {len(vector)}")
            print(f"First 5 values: {vector[:5]}")
            print(f"L2 norm: {np.linalg.norm(vector):.4f}")

            # Show original text if available
            entry = metadata["entries"].get(str(id), {})
            text = entry.get("text", "Unknown")
            timestamp = entry.get("timestamp", "Unknown")

            print(f'Original text: "{text}"')
            print(f"Stored on: {timestamp}")

            return vector
        else:
            print(f"Failed to retrieve vector {id}")
            return None
    except Exception as e:
        print(f"Error in retrieve_vector: {e}")
        import traceback

        traceback.print_exc()
        return None


# Find closest vectors
def find_closest(store, text, metadata, k=5):
    embedding = get_embedding(text)
    if embedding is None:
        return

    try:
        start_time = time.time()
        results = store.find_similar_vectors(embedding, k)
        elapsed = time.time() - start_time

        print(f"Search completed in {elapsed*1000:.2f}ms")

        if results:
            print(f'\nTop {len(results)} matches for: "{text}"')
            print("-" * 80)
            # Sort by similarity (highest first)
            for id, similarity in sorted(results, key=lambda x: x[1], reverse=True):
                # Try to parse metadata from string if it's JSON
                try:
                    entry = metadata["entries"].get(str(id), {})
                    match_text = entry.get("text", "Unknown")
                    timestamp = entry.get("timestamp", "Unknown")
                except:
                    # Fallback
                    match_text = "Unknown"
                    timestamp = "Unknown"

                print(f'ID: {id:4} | Score: {similarity:.4f} | "{match_text}"')
                print(f"      Stored: {timestamp}")
                print("-" * 80)
        else:
            print("No results found")
    except Exception as e:
        print(f"Error in find_closest: {e}")
        import traceback

        traceback.print_exc()
        return


# Interactive shell
#def interactive_shell():
#    print("\n" + "=" * 80)
#    print("VECTOR CLUSTER STORE CLI - Ollama Embedding Search".center(80))
#    print("=" * 80)
#
#    store = init_vector_store()
#    metadata = load_metadata()
#
#    # Check if metadata has the right vector dimension
#    if "vector_dim" in metadata and metadata["vector_dim"] != VECTOR_DIM:
#        print(
#            f"Warning: Metadata vector dimension ({metadata['vector_dim']}) doesn't match current ({VECTOR_DIM})"
#        )
#        print(
#            "This might cause issues with retrieving vectors. Consider creating a new store file."
#        )
#
#    print(f"\nUsing model: {EMBEDDING_MODEL}")
#    print(f"Vector dimension: {VECTOR_DIM}")
#    print(f"Storage file: {DEVICE_PATH}")
#    print(f"Next ID: {metadata['next_id']}")
#    print(f"Stored entries: {len(metadata['entries'])}")
#
#    print("\nMenu Options:")
#    print("1. Store vectors from text")
#    print("2. Retrieve vector by ID")
#    print("3. Find closest matches")
#    print("4. Perform maintenance")
#    print("5. Save index to file")
#    print("6. Load index from file")
#    print("7. Print store info")
#    print("8. Exit")
#
#    while True:
#        try:
#            choice = input("\nEnter your choice (1-8): ")
#
#            if choice == "1":
#                print(
#                    "\nStore Mode: Enter text to embed (or 'done' to return to menu):"
#                )
#                while True:
#                    text = input("> ")
#                    if text.lower() == "done":
#                        break
#                    store_text(store, text, metadata)
#
#            elif choice == "2":
#                id_str = input("\nEnter vector ID to retrieve: ")
#                try:
#                    id = int(id_str)
#                    retrieve_vector(store, id, metadata)
#                except ValueError:
#                    print("Please enter a valid numeric ID")
#
#            elif choice == "3":
#                text = input("\nEnter text to find similar embeddings: ")
#                k_str = input("Number of results to return (default 5): ")
#                try:
#                    k = int(k_str) if k_str.strip() else 5
#                except ValueError:
#                    k = 5
#                find_closest(store, text, metadata, k)
#
#            elif choice == "4":
#                print("\nPerforming maintenance...")
#                if store.perform_maintenance():
#                    print("Maintenance completed successfully")
#                else:
#                    print("Maintenance failed")
#
#            elif choice == "5":
#                filename = input(
#                    "\nEnter filename to save index (default: vector_store_index): "
#                )
#                if not filename.strip():
#                    filename = "vector_store_index"
#                if store.save_index(filename):
#                    print(f"Index saved successfully to {filename}")
#                else:
#                    print("Failed to save index")
#
#            elif choice == "6":
#                filename = input(
#                    "\nEnter filename to load index from (default: vector_store_index): "
#                )
#                if not filename.strip():
#                    filename = "vector_store_index"
#                if store.load_index(filename):
#                    print(f"Index loaded successfully from {filename}")
#                else:
#                    print("Failed to load index")
#
#            elif choice == "7":
#                store.print_store_info()
#                clusters = input(
#                    "\nEnter cluster ID to see details (or press Enter to skip): "
#                )
#                if clusters.strip():
#                    try:
#                        cluster_id = int(clusters)
#                        store.print_cluster_info(cluster_id)
#                    except ValueError:
#                        print("Please enter a valid numeric ID")
#
#            elif choice == "8":
#                print("\nExiting...")
#                break
#
#            else:
#                print("Invalid choice, please enter a number between 1 and 8")
#
#        except KeyboardInterrupt:
#            print("\nExiting...")
#            break
#        except Exception as e:
#            print(f"Error: {e}")
#            import traceback
#
#            traceback.print_exc()


import requests
from typing import Dict, Any, List, Optional, Tuple, Union
# Initialize repair tool
repair_tool = EnhancedJSONRepair()

def extract_and_parse_json_blocks(text: str) -> List[Dict[str, Any]]:
    """
    Extract and parse JSON blocks from raw text, with comprehensive error handling
    
    Args:
        text: The text containing JSON blocks
        
    Returns:
        List of parsed JSON objects
    """
    json_objects = []
    
    
    # Find all JSON-like blocks in the text
    # We'll look for blocks starting with raw_text: followed by JSON content
    raw_text_blocks = re.findall(r'\s*(\{[\s\S]*?\})\n', text)
    
    for block in raw_text_blocks:
        try:
            # Try direct parsing first
            try:
                json_obj = json.loads(block)
                json_objects.append(json_obj)
                continue
            except json.JSONDecodeError:
                pass
            
            # If direct parsing fails, try repair
            repaired_obj, success, error = repair_tool.repair_json(block)
            
            if success:
                json_objects.append(repaired_obj)
            else:
                print(f"Failed to repair JSON: {error}")
                print(f"Block: {block[:100]}...")
        except Exception as e:
            print(f"Error processing JSON block: {str(e)}")
    
    return json_objects



def createIndexText(data):

    if type(data) == dict:
        for key, value in data.items():
            createIndexText(value)
    elif type(data) == list:
        for item in data:
            createIndexText(item)
    else:
        code = extract_and_parse_json_blocks(data)
        if len(code) > 0:
            for item in code:
                if "error" in item:
                    #print(f"error: {item['error']}")
                    print(f"raw_text: {item['raw_text']}")
                else:
                    for k, v in item.items():
                        print(f"{k} : {v}")
        else:
            pass

def junk(payload):

    name=payload.get('document')
    page=payload.get('page')
    format_=payload.get('format')
    data=payload.get('data')


def store_vec(data, name, page):
    with open (f"./repaired/{name}_{page}.json", 'w') as fout:
        fout.write (json.dumps(data))
    print (f'{name} {page}') 
    store_text(store, json.dumps(data), metadata)
    #print (f"{data}\n\n")

    
def extractr(data):
    try:
        results = extract_and_parse_json_blocks(data)
        return results
    except:
        pass
        #print (f"ERROR:::{data.get('data')}")
    return None

if __name__ == "__main__":
    with sys.stdin as f:
        for idx, line in enumerate(f):
            line = line.strip()
            with open(f"{line}", "r") as f:
                data = json.load(f)
                if type(data) is dict:
                    name=line.split('/')[-1].split('.')[0]
                    page="1"
                    results=extractr(json.dumps(data))
                    store_vec (data, name, page)
                elif type(data) is list:
                    for i, sample in enumerate(data):
                        name=sample.get('document')
                        page=sample.get('page')
                        results=extractr(sample.get('data'))
                        store_vec (results, name, page)
                else:
                    pass
                    print (data)
                    continue
