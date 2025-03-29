#!/usr/bin/env python3
"""
Extract, repair, and save JSON blocks from a text file.

This script processes text files containing raw_text: prefixed JSON blocks,
repairs any malformed JSON, and saves each block as a separate file.
"""

import os
import sys
import re
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple
from enhanced_json_repair import EnhancedJSONRepair  # Import from the enhanced JSON repair module

def extract_json_blocks(file_path: str) -> List[str]:
    """
    Extract raw_text: blocks from a file
    
    Args:
        file_path: Path to the file containing raw_text: blocks
        
    Returns:
        List of raw JSON block strings
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all raw_text: blocks
    blocks = re.findall(r'raw_text:\s*([\s\S]*?)(?=raw_text:|$)', content)
    
    # Clean up blocks
    cleaned_blocks = []
    for block in blocks:
        # Trim whitespace and ensure it looks like JSON
        block = block.strip()
        if block.startswith('{') and block.endswith('}'):
            cleaned_blocks.append(block)
    
    return cleaned_blocks

def process_json_blocks(blocks: List[str], 
                        repair_tool: Optional[EnhancedJSONRepair] = None, 
                        debug: bool = False) -> List[Dict[str, Any]]:
    """
    Process and repair JSON blocks
    
    Args:
        blocks: List of raw JSON block strings
        repair_tool: The EnhancedJSONRepair instance to use (creates a new one if None)
        debug: Whether to print debug information
        
    Returns:
        List of parsed JSON objects
    """
    if repair_tool is None:
        repair_tool = EnhancedJSONRepair()
    
    results = []
    
    for i, block in enumerate(blocks):
        if debug:
            print(f"Processing block {i+1}/{len(blocks)}")
        
        try:
            # Try direct parsing
            try:
                json_obj = json.loads(block)
                results.append(json_obj)
                if debug:
                    print(f"  Block {i+1} parsed successfully without repair")
                continue
            except json.JSONDecodeError as e:
                if debug:
                    print(f"  Block {i+1} failed to parse: {str(e)}")
            
            # Try repair
            repaired_obj, success, error = repair_tool.repair_json(block)
            
            if success:
                results.append(repaired_obj)
                if debug:
                    print(f"  Block {i+1} repaired successfully")
            else:
                if debug:
                    print(f"  Block {i+1} repair failed: {error}")
                    print(f"  First 100 chars: {block[:100]}...")
        
        except Exception as e:
            if debug:
                print(f"  Error processing block {i+1}: {str(e)}")
    
    return results

def save_json_objects(objects: List[Dict[str, Any]], output_dir: str, 
                     prefix: str = "json_block_") -> None:
    """
    Save JSON objects to files
    
    Args:
        objects: List of JSON objects
        output_dir: Directory to save files
        prefix: Prefix for file names
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, obj in enumerate(objects):
        file_path = os.path.join(output_dir, f"{prefix}{i+1}.json")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=2)

def extract_json_from_file(file_path: str, output_dir: str, 
                          debug: bool = False) -> None:
    """
    Extract, repair, and save JSON blocks from a file
    
    Args:
        file_path: Path to the file containing raw_text: blocks
        output_dir: Directory to save repaired JSON files
        debug: Whether to print debug information
    """
    # Extract blocks
    blocks = extract_json_blocks(file_path)
    
    if debug:
        print(f"Found {len(blocks)} potential JSON blocks in {file_path}")
    
    if not blocks:
        if debug:
            print(f"No JSON blocks found in {file_path}")
        return
    
    # Initialize repair tool
    repair_tool = EnhancedJSONRepair()
    
    # Process blocks
    json_objects = process_json_blocks(blocks, repair_tool, debug)
    
    # Create output directory based on file name
    file_name = os.path.basename(file_path)
    file_prefix = os.path.splitext(file_name)[0]
    specific_output_dir = os.path.join(output_dir, file_prefix)
    
    # Save results
    save_json_objects(json_objects, specific_output_dir)
    
    if debug:
        print(f"Saved {len(json_objects)} JSON files to {specific_output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Extract and repair JSON blocks from files")
    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("--output", "-o", default="extracted_json", 
                       help="Output directory (default: ./extracted_json)")
    parser.add_argument("--recursive", "-r", action="store_true", 
                       help="Process directories recursively")
    parser.add_argument("--debug", "-d", action="store_true", 
                       help="Print debug information")
    
    args = parser.parse_args()
    
    # Check if input is a file or directory
    if os.path.isfile(args.input):
        # Process a single file
        extract_json_from_file(args.input, args.output, args.debug)
    elif os.path.isdir(args.input):
        # Process a directory
        for root, dirs, files in os.walk(args.input):
            if not args.recursive and root != args.input:
                continue
            
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    extract_json_from_file(file_path, args.output, args.debug)
                except Exception as e:
                    if args.debug:
                        print(f"Error processing {file_path}: {str(e)}")
    else:
        print(f"Error: {args.input} is not a valid file or directory.")
        sys.exit(1)

if __name__ == "__main__":
    main()
