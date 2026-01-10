#!/usr/bin/env python3
"""
fastcomp - Fast Vector Comparison Tool

Compare text embeddings using Ollama API.
Reads text from stdin, computes embeddings, and outputs distances.
"""
import sys
import argparse
import time
from typing import List, Optional

import numpy as np

try:
    import requests
except ImportError:
    requests = None

# Configuration
OLLAMA_API_URL = "http://127.0.0.1:11434/api/embed"
EMBEDDING_MODEL = "nomic-embed-text"


def get_embedding(text: str, model: str = EMBEDDING_MODEL, api_url: str = OLLAMA_API_URL) -> Optional[np.ndarray]:
    """Get embedding from Ollama API."""
    if requests is None:
        print("Error: requests library not installed. Run: pip install requests", file=sys.stderr)
        return None

    try:
        response = requests.post(
            api_url,
            json={"model": model, "input": text},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()

        data = response.json()
        if "embeddings" not in data or not data["embeddings"]:
            print("Error: Invalid response format - missing embeddings", file=sys.stderr)
            return None

        return np.array(data["embeddings"][0], dtype=np.float32)

    except requests.exceptions.ConnectionError:
        print(f"Error: Cannot connect to Ollama at {api_url}", file=sys.stderr)
        print("Make sure Ollama is running: ollama serve", file=sys.stderr)
        return None
    except requests.exceptions.Timeout:
        print("Error: Request timed out", file=sys.stderr)
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error: HTTP request failed: {e}", file=sys.stderr)
        return None
    except (KeyError, IndexError, ValueError) as e:
        print(f"Error: Failed to parse response: {e}", file=sys.stderr)
        return None


def cosine_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate cosine distance (1 - cosine similarity) between two vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 1.0  # Maximum distance for zero vectors

    similarity = np.dot(v1, v2) / (norm1 * norm2)
    return 1.0 - similarity


def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate Euclidean distance between two vectors."""
    return float(np.linalg.norm(v1 - v2))


def compare_texts(texts: List[str], metric: str = "cosine",
                  model: str = EMBEDDING_MODEL, api_url: str = OLLAMA_API_URL) -> Optional[List[float]]:
    """
    Compare texts against the first text (basis).

    Args:
        texts: List of texts. First is basis, rest are compared against it.
        metric: Distance metric - "cosine" or "euclidean"
        model: Ollama embedding model name
        api_url: Ollama API URL

    Returns:
        List of distances, or None on error
    """
    if len(texts) < 2:
        print("Error: Need at least 2 texts to compare", file=sys.stderr)
        return None

    # Get basis embedding
    basis = get_embedding(texts[0], model, api_url)
    if basis is None:
        print("Error: Failed to get embedding for basis text", file=sys.stderr)
        return None

    # Select distance function
    distance_fn = cosine_distance if metric == "cosine" else euclidean_distance

    # Compare against each text
    distances = []
    for i, text in enumerate(texts[1:], start=2):
        embedding = get_embedding(text, model, api_url)
        if embedding is None:
            print(f"Error: Failed to get embedding for text {i}", file=sys.stderr)
            return None

        if len(embedding) != len(basis):
            print(f"Error: Dimension mismatch at text {i}", file=sys.stderr)
            return None

        distances.append(distance_fn(basis, embedding))

    return distances


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fast vector comparison tool for text embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input format:
  Reads text from stdin, one line per text to compare.
  First line is the basis vector (v0).
  Subsequent lines are compared against v0.

Output:
  Prints distance values to stdout, one per line.

Example:
  echo -e 'Michigan\\nDetroit\\nChicago\\nCalifornia' | fastcomp

  printf 'cat\\ndog\\ncar\\n' | fastcomp -m euclidean
"""
    )
    parser.add_argument(
        "-m", "--metric",
        choices=["cosine", "euclidean"],
        default="cosine",
        help="Distance metric (default: cosine)"
    )
    parser.add_argument(
        "--model",
        default=EMBEDDING_MODEL,
        help=f"Ollama embedding model (default: {EMBEDDING_MODEL})"
    )
    parser.add_argument(
        "--api-url",
        default=OLLAMA_API_URL,
        help=f"Ollama API URL (default: {OLLAMA_API_URL})"
    )

    args = parser.parse_args()

    # Check for requests library
    if requests is None:
        print("Error: requests library required. Install with: pip install requests", file=sys.stderr)
        sys.exit(1)

    # Read input
    texts = [line.strip() for line in sys.stdin if line.strip()]

    if not texts:
        print("Error: No input text provided", file=sys.stderr)
        sys.exit(1)

    if len(texts) < 2:
        print("Error: Need at least 2 texts to compare (basis + 1 comparison)", file=sys.stderr)
        sys.exit(1)

    # Compare texts
    start = time.time()
    distances = compare_texts(texts, args.metric, args.model, args.api_url)
    elapsed = (time.time() - start) * 1000

    if distances is None:
        sys.exit(1)

    # Output distances
    for d in distances:
        print(f"{d:.6f}")

    # Timing info to stderr
    print(f"Processed {len(texts)} texts in {elapsed:.0f}ms", file=sys.stderr)


if __name__ == "__main__":
    main()
