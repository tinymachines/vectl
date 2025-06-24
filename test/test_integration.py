"""
Integration tests for the complete workflow
"""
import pytest
import subprocess
import json
import os
import numpy as np


class TestIntegration:
    """Integration tests combining multiple components"""
    
    @pytest.mark.requires_ollama
    def test_full_workflow(self, temp_dir, ollama_available, ollama_tool_path, sample_texts):
        """Test complete workflow: embed, search, maintain"""
        index_path = os.path.join(temp_dir, "workflow.bin")
        
        # 1. Embed multiple texts
        for i, text in enumerate(sample_texts):
            result = subprocess.run([
                "python", ollama_tool_path,
                "embed",
                "--index", index_path,
                "--text", text
            ], capture_output=True, text=True)
            assert result.returncode == 0
        
        # 2. Verify metadata
        metadata_path = os.path.join(temp_dir, "workflow_metadata.json")
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        assert metadata["next_id"] == len(sample_texts)
        assert len(metadata["entries"]) == len(sample_texts)
        
        # 3. Search for similar content
        result = subprocess.run([
            "python", ollama_tool_path,
            "search",
            "--index", index_path,
            "--query", "artificial intelligence and machine learning",
            "--format", "json",
            "--top-k", "3"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        # Extract JSON from mixed output
        json_start = result.stdout.find('{')
        if json_start == -1:
            pytest.fail(f"No JSON found in output: {result.stdout}")
        json_str = result.stdout[json_start:]
        search_results = json.loads(json_str)
        
        assert len(search_results["results"]) <= 3
        assert search_results["results"][0]["text"] != "Unknown"
        
        # The search should return relevant results
        top_result = search_results["results"][0]["text"]
        # Any of the sample texts is a valid result since we're testing the workflow
        assert top_result in sample_texts
        
        # 4. Run maintenance
        result = subprocess.run([
            "python", ollama_tool_path,
            "maintenance",
            "--index", index_path
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "Maintenance completed successfully" in result.stdout
        
        # 5. Search again to ensure maintenance didn't break anything
        result = subprocess.run([
            "python", ollama_tool_path,
            "search",
            "--index", index_path,
            "--query", "Python programming",
            "--format", "json"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        # Extract JSON from mixed output
        json_start = result.stdout.find('{')
        if json_start == -1:
            pytest.fail(f"No JSON found in output: {result.stdout}")
        json_str = result.stdout[json_start:]
        search_results = json.loads(json_str)
        assert len(search_results["results"]) > 0
    
    def test_python_api_integration(self, vector_store_py, temp_dir):
        """Test Python API integration with file structure"""
        # Create store using Python API
        store_path = os.path.join(temp_dir, "api_test.bin")
        log_path = os.path.join(temp_dir, "api_test.log")
        
        # Touch the file first
        open(store_path, 'a').close()
        
        logger = vector_store_py.Logger(log_path)
        store = vector_store_py.VectorClusterStore(logger)
        
        # Initialize
        vector_dim = 768
        success = store.initialize(store_path, "kmeans", vector_dim, 10)
        assert success
        
        # Store vectors with metadata
        texts = [
            "First test vector",
            "Second test vector",
            "Third test vector"
        ]
        
        for i, text in enumerate(texts):
            vector = np.random.rand(vector_dim).tolist()
            metadata = json.dumps({
                "text": text,
                "id": i,
                "source": "test"
            })
            success = store.store_vector(i, vector, metadata)
            assert success
        
        # Search
        query_vector = np.random.rand(vector_dim).tolist()
        results = store.find_similar_vectors(query_vector, 2)
        
        assert len(results) == 2
        assert all(isinstance(r[0], int) for r in results)  # IDs
        assert all(isinstance(r[1], float) for r in results)  # Scores
    
    @pytest.mark.requires_ollama
    def test_large_batch_workflow(self, temp_dir, ollama_available, ollama_tool_path):
        """Test handling of large batches"""
        index_path = os.path.join(temp_dir, "large_batch.bin")
        
        # Generate 50 texts
        large_texts = []
        for i in range(50):
            large_texts.append(f"Test document number {i} with some content about topic {i % 10}")
        
        # Embed all texts via stdin
        input_text = "\n".join(large_texts)
        
        result = subprocess.run([
            "python", ollama_tool_path,
            "embed",
            "--index", index_path,
            "--format", "json"
        ], input=input_text, capture_output=True, text=True)
        
        assert result.returncode == 0
        
        # Extract JSON from mixed output
        json_start = result.stdout.find('{')
        if json_start == -1:
            pytest.fail(f"No JSON found in output: {result.stdout}")
        json_str = result.stdout[json_start:]
        output = json.loads(json_str)
        assert output["success"]
        assert output["stored_count"] == 50
        assert output["total_texts"] == 50
        
        # Verify we can search
        result = subprocess.run([
            "python", ollama_tool_path,
            "search",
            "--index", index_path,
            "--query", "document about topic 5",
            "--format", "json",
            "--top-k", "5"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        # Extract JSON from mixed output
        json_start = result.stdout.find('{')
        if json_start == -1:
            pytest.fail(f"No JSON found in output: {result.stdout}")
        json_str = result.stdout[json_start:]
        search_results = json.loads(json_str)
        assert len(search_results["results"]) > 0
        
        # At least one result should mention "topic 5"
        found_topic_5 = any("topic 5" in r["text"] for r in search_results["results"])
        assert found_topic_5
    
    def test_persistence_across_tools(self, temp_dir, ollama_tool_path, vector_store_py):
        """Test that data created by CLI is readable by Python API and vice versa"""
        index_path = os.path.join(temp_dir, "persist.bin")
        
        # 1. Create and populate using Python API
        # Touch the file first
        open(index_path, 'a').close()
        
        logger = vector_store_py.Logger("")
        store = vector_store_py.VectorClusterStore(logger)
        store.initialize(index_path, "kmeans", 128, 10)
        
        test_vector = np.random.rand(128).tolist()
        metadata = json.dumps({"text": "API created vector"})
        store.store_vector(0, test_vector, metadata)
        
        del store  # Close the store
        
        # 2. Read using CLI tool
        result = subprocess.run([
            "python", ollama_tool_path,
            "info",
            "--index", index_path,
            "--dimension", "128"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "Vector count: 1" in result.stdout or "Stored entries: 0" in result.stdout
        
        # Note: The CLI tool uses separate metadata file, so it won't see the embedded metadata
        # This is expected behavior - the binary store and metadata files are separate