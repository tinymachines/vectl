"""
Tests for the ollama_tool.py command-line interface
"""
import pytest
import subprocess
import json
import os
import tempfile


class TestOllamaTool:
    """Test the ollama_tool.py CLI"""
    
    def run_ollama_tool(self, args, input_text=None):
        """Helper to run ollama_tool.py with arguments"""
        cmd = ["python", self.ollama_tool_path] + args
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            input=input_text
        )
        return result
    
    @pytest.fixture(autouse=True)
    def setup(self, ollama_tool_path):
        """Setup for each test"""
        self.ollama_tool_path = ollama_tool_path
    
    def test_help(self):
        """Test help output"""
        result = self.run_ollama_tool(["-h"])
        assert result.returncode == 0
        assert "Ollama Vector Store Tool" in result.stdout
        assert "embed" in result.stdout
        assert "search" in result.stdout
        assert "maintenance" in result.stdout
        assert "info" in result.stdout
    
    def test_file_creation_default_location(self, temp_dir):
        """Test that files are created in default location"""
        # Change to temp directory
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Run info command (should create files)
            result = self.run_ollama_tool(["info"])
            assert result.returncode == 0
            
            # Check files were created
            assert os.path.exists("vector_store.bin")
            assert os.path.exists("vector_store.log")
        finally:
            os.chdir(original_dir)
    
    def test_file_creation_custom_location(self, temp_dir):
        """Test that files are created in custom location"""
        custom_path = os.path.join(temp_dir, "subdir", "custom.bin")
        
        # Run info command with custom path
        result = self.run_ollama_tool(["info", "--index", custom_path])
        assert result.returncode == 0
        
        # Check files were created
        assert os.path.exists(custom_path)
        assert os.path.exists(os.path.join(temp_dir, "subdir", "custom.log"))
    
    @pytest.mark.requires_ollama
    def test_embed_single_text(self, temp_dir, ollama_available):
        """Test embedding a single text"""
        index_path = os.path.join(temp_dir, "test.bin")
        
        # Embed text
        result = self.run_ollama_tool([
            "embed",
            "--index", index_path,
            "--text", "Test embedding",
            "--format", "json"
        ])
        
        assert result.returncode == 0
        
        # Parse JSON output (extract JSON from mixed output)
        # Find the JSON part starting with '{'
        json_start = result.stdout.find('{')
        if json_start == -1:
            pytest.fail(f"No JSON found in output: {result.stdout}")
        json_str = result.stdout[json_start:]
        output = json.loads(json_str)
        assert output["success"]
        assert output["stored_count"] == 1
        assert output["total_texts"] == 1
        
        # Check metadata file
        metadata_path = os.path.join(temp_dir, "test_metadata.json")
        assert os.path.exists(metadata_path)
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        assert metadata["next_id"] == 1
        assert "0" in metadata["entries"]
        assert metadata["entries"]["0"]["text"] == "Test embedding"
    
    @pytest.mark.requires_ollama
    def test_embed_batch_mode(self, temp_dir, ollama_available):
        """Test batch embedding from stdin"""
        index_path = os.path.join(temp_dir, "batch.bin")
        
        # Prepare input texts
        input_texts = "First text\nSecond text\nThird text\n"
        
        # Embed texts
        result = self.run_ollama_tool([
            "embed",
            "--index", index_path,
            "--format", "json"
        ], input_text=input_texts)
        
        assert result.returncode == 0
        
        # Parse JSON output (extract JSON from mixed output)
        # Find the JSON part starting with '{'
        json_start = result.stdout.find('{')
        if json_start == -1:
            pytest.fail(f"No JSON found in output: {result.stdout}")
        json_str = result.stdout[json_start:]
        output = json.loads(json_str)
        assert output["success"]
        assert output["stored_count"] == 3
        assert output["total_texts"] == 3
        
        # Check metadata
        metadata_path = os.path.join(temp_dir, "batch_metadata.json")
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        assert metadata["next_id"] == 3
        assert len(metadata["entries"]) == 3
        assert metadata["entries"]["0"]["text"] == "First text"
        assert metadata["entries"]["1"]["text"] == "Second text"
        assert metadata["entries"]["2"]["text"] == "Third text"
    
    @pytest.mark.requires_ollama
    def test_search_json_format(self, temp_dir, ollama_available):
        """Test search with JSON output"""
        index_path = os.path.join(temp_dir, "search.bin")
        
        # First embed some texts
        self.run_ollama_tool([
            "embed",
            "--index", index_path,
            "--text", "Machine learning and AI"
        ])
        
        self.run_ollama_tool([
            "embed",
            "--index", index_path,
            "--text", "Deep learning networks"
        ])
        
        # Search
        result = self.run_ollama_tool([
            "search",
            "--index", index_path,
            "--query", "artificial intelligence",
            "--format", "json",
            "--top-k", "2"
        ])
        
        assert result.returncode == 0
        
        # Parse JSON output (extract JSON from mixed output)
        # Find the JSON part starting with '{'
        json_start = result.stdout.find('{')
        if json_start == -1:
            pytest.fail(f"No JSON found in output: {result.stdout}")
        json_str = result.stdout[json_start:]
        output = json.loads(json_str)
        assert "query" in output
        assert output["query"] == "artificial intelligence"
        assert "results" in output
        assert len(output["results"]) <= 2
        
        # Check result structure
        if output["results"]:
            result = output["results"][0]
            assert "id" in result
            assert "score" in result
            assert "text" in result
            assert "timestamp" in result
            assert "model" in result
            # Text should not be "Unknown"
            assert result["text"] != "Unknown"
    
    @pytest.mark.requires_ollama
    def test_search_text_format(self, temp_dir, ollama_available):
        """Test search with text/table output"""
        index_path = os.path.join(temp_dir, "search_text.bin")
        
        # Embed a text
        self.run_ollama_tool([
            "embed",
            "--index", index_path,
            "--text", "Vector databases for similarity search"
        ])
        
        # Search
        result = self.run_ollama_tool([
            "search",
            "--index", index_path,
            "--query", "vector search",
            "--top-k", "1"
        ])
        
        assert result.returncode == 0
        assert "Top 1 matches for:" in result.stdout
        assert "Vector databases for similarity search" in result.stdout
        assert "Score" in result.stdout
    
    def test_maintenance_mode(self, temp_dir):
        """Test maintenance mode"""
        index_path = os.path.join(temp_dir, "maint.bin")
        
        # Create a store first
        self.run_ollama_tool(["info", "--index", index_path])
        
        # Run maintenance
        result = self.run_ollama_tool([
            "maintenance",
            "--index", index_path
        ])
        
        assert result.returncode == 0
        assert "Maintenance completed successfully" in result.stdout
    
    def test_info_mode(self, temp_dir):
        """Test info mode"""
        index_path = os.path.join(temp_dir, "info.bin")
        
        # Run info
        result = self.run_ollama_tool([
            "info",
            "--index", index_path
        ])
        
        assert result.returncode == 0
        assert "Vector Store Information:" in result.stdout
        assert f"Index path: {index_path}" in result.stdout
        assert "Vector dimension: 768" in result.stdout
        assert "Number of clusters: 10" in result.stdout
    
    def test_custom_parameters(self, temp_dir):
        """Test custom parameters"""
        index_path = os.path.join(temp_dir, "custom.bin")
        
        # Run with custom parameters
        result = self.run_ollama_tool([
            "info",
            "--index", index_path,
            "--dimension", "512",
            "--clusters", "20",
            "--model", "custom-model"
        ])
        
        assert result.returncode == 0
        assert "Vector dimension: 512" in result.stdout
        assert "Number of clusters: 20" in result.stdout
        assert "Embedding model: custom-model" in result.stdout
    
    def test_verbose_mode(self, temp_dir):
        """Test verbose output"""
        index_path = os.path.join(temp_dir, "verbose.bin")
        
        # Run with verbose
        result = self.run_ollama_tool([
            "info",
            "--index", index_path,
            "--verbose"
        ])
        
        assert result.returncode == 0
        assert "Vector store initialized successfully" in result.stdout
        assert f"Device path: {index_path}" in result.stdout
    
    def test_empty_input_handling(self, temp_dir):
        """Test handling of empty input"""
        index_path = os.path.join(temp_dir, "empty.bin")
        
        # Try to embed with empty stdin
        result = self.run_ollama_tool([
            "embed",
            "--index", index_path
        ], input_text="")
        
        assert result.returncode == 1
        assert "No text provided for embedding" in result.stdout