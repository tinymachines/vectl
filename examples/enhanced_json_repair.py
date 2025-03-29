import json
import re
import requests
from typing import Dict, Any, List, Optional, Tuple, Union


class EnhancedJSONRepair:
    def __init__(self, 
                 ollama_url: str = "https://ollama.meatball.ai/api/generate",
                 model_name: str = "gemma3:12b",
                 enable_llm_repair: bool = True):
        """
        Initialize the enhanced JSON repair utility
        
        Args:
            ollama_url: URL for Ollama API
            model_name: Model to use for repair
            enable_llm_repair: Whether to use LLM for repair (if False, will only use basic repair)
        """
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.enable_llm_repair = enable_llm_repair
    
    def repair_json(self, json_str: str) -> Tuple[Dict[str, Any], bool, Optional[str]]:
        """
        Attempt to repair and parse a potentially malformed JSON string
        
        Args:
            json_str: The JSON string to repair
            
        Returns:
            Tuple of (parsed JSON object, success flag, error message if any)
        """
        # Preprocessing - handle raw_text prefix if present
        if json_str.startswith("raw_text:"):
            json_str = json_str.replace("raw_text:", "", 1).strip()
        
        # First try direct parsing
        try:
            result = json.loads(json_str)
            return result, True, None
        except json.JSONDecodeError as e:
            pass
        
        # Try basic repair techniques
        repaired_json = self._basic_repair(json_str)
        try:
            result = json.loads(repaired_json)
            return result, True, None
        except json.JSONDecodeError as e:
            # Try structural repair for more complex issues
            try:
                structurally_repaired = self._structural_repair(json_str)
                result = json.loads(structurally_repaired)
                return result, True, None
            except json.JSONDecodeError:
                if not self.enable_llm_repair:
                    return {}, False, f"JSON parsing failed: {str(e)}"
                
                # If basic repair failed, try LLM-based repair
                try:
                    llm_repaired = self._llm_repair(json_str, str(e))
                    result = json.loads(llm_repaired)
                    return result, True, None
                except json.JSONDecodeError as e2:
                    # Try a complete rebuild of the JSON structure
                    try:
                        rebuilt_json = self._rebuild_json_from_patterns(json_str)
                        result = json.loads(rebuilt_json)
                        return result, True, None
                    except Exception:
                        return {}, False, f"LLM repair failed: {str(e2)}"
                except Exception as e3:
                    return {}, False, f"LLM API error: {str(e3)}"
    
    def _basic_repair(self, json_str: str) -> str:
        """
        Apply basic JSON repair techniques
        
        Args:
            json_str: The JSON string to repair
            
        Returns:
            Repaired JSON string
        """
        # Remove comments first (JSON doesn't support comments)
        json_str = self._remove_comments(json_str)
        
        # Remove escaped newlines and replace with actual newlines
        json_str = json_str.replace('\\n', '\n')
        
        # Remove escaped quotes and replace with actual quotes
        json_str = json_str.replace('\\"', '"')
        
        # Try to fix common section numbering pattern issues
        # Look for },\n    \"title\": pattern without a section declaration
        pattern = r'},\s*"title":'
        if re.search(pattern, json_str):
            section_counter = 1
            last_found = 0
            
            while True:
                # Look for the next section
                current_section = f'"section_{section_counter}":'
                next_section = f'"section_{section_counter + 1}":'
                
                # Find positions
                current_pos = json_str.find(current_section, last_found)
                if current_pos == -1:
                    break
                
                # Update last found position
                last_found = current_pos + len(current_section)
                
                # Check if next section exists
                next_pos = json_str.find(next_section, last_found)
                
                # Look for potential missing section
                pattern = r'},\s*"title":'
                missing_section_match = re.search(pattern, json_str[last_found:next_pos if next_pos != -1 else None])
                
                if missing_section_match:
                    # Calculate the position in the original string
                    match_pos = last_found + missing_section_match.start()
                    
                    # Insert the missing section declaration
                    section_id = section_counter + 1
                    replacement = f'}},\n  "section_{section_id}": {{\n    "title":'
                    json_str = json_str[:match_pos] + replacement + json_str[match_pos+missing_section_match.end()-len('"title":'):]
                    
                    # Skip ahead to the next section
                    section_counter = section_id
                
                section_counter += 1
        
        # Fix missing commas in lists/objects
        json_str = self._add_missing_commas(json_str)
        
        # Fix trailing commas in objects
        json_str = self._fix_trailing_commas(json_str)
        
        # Add missing values for keys with no values
        json_str = self._add_missing_values(json_str)
        
        # Check for missing opening braces in sections
        pattern = r'"section_\d+":\s*"'
        for match in re.finditer(pattern, json_str):
            pos = match.end() - 1
            json_str = json_str[:pos] + '{' + json_str[pos:]
        
        # Check for missing closing braces at the end
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        
        if open_braces > close_braces:
            json_str = json_str.rstrip() + '\n' + ('}' * (open_braces - close_braces))
        
        return json_str

    def _remove_comments(self, json_str: str) -> str:
        """Remove comments from JSON string"""
        # Remove single-line comments (// ...)
        json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
        
        # Remove multi-line comments (/* ... */)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        
        return json_str
    
    def _add_missing_commas(self, json_str: str) -> str:
        """Add missing commas between objects and array elements"""
        # Look for missing commas between array elements
        json_str = re.sub(r'}\s*{', '}, {', json_str)
        json_str = re.sub(r'"\s*{', '", {', json_str)
        json_str = re.sub(r'}\s*"', '}, "', json_str)
        
        return json_str
    
    def _fix_trailing_commas(self, json_str: str) -> str:
        """
        Fix trailing commas in objects and arrays
        
        These are invalid in standard JSON but commonly appear in the data
        """
        # Fix trailing commas in objects
        json_str = re.sub(r',\s*}', '}', json_str)
        
        # Fix trailing commas in arrays
        json_str = re.sub(r',\s*]', ']', json_str)
        
        return json_str
    
    def _add_missing_values(self, json_str: str) -> str:
        """Add empty values for keys with no values"""
        # Pattern for keys with no values in an object (ending with comma or closing brace)
        # This is a common pattern in your examples: {"title": "Something",}
        pattern = r'"([^"]+)":\s*,|"([^"]+)":\s*}'
        
        # Add empty string for missing values with comma
        json_str = re.sub(r'"([^"]+)":\s*,', r'"\1": "",', json_str)
        
        # Add empty string for missing values at end of object
        json_str = re.sub(r'"([^"]+)":\s*}', r'"\1": ""}', json_str)
        
        return json_str
    
    def _structural_repair(self, json_str: str) -> str:
        """
        Apply more advanced structural repairs to the JSON
        
        This handles cases where the JSON structure is significantly damaged
        """
        # Fix truncated arrays or objects
        json_str = self._fix_truncated_structures(json_str)
        
        # Handle incomplete or malformed key-value pairs
        json_str = self._fix_malformed_key_values(json_str)
        
        # Ensure all keys are properly quoted
        json_str = self._ensure_quoted_keys(json_str)
        
        # Remove any hanging fields at the document level (they're not within an object)
        json_str = self._remove_hanging_fields(json_str)
        
        return json_str
    
    def _fix_truncated_structures(self, json_str: str) -> str:
        """Fix truncated arrays or objects indicated by ellipses or comments"""
        # Replace "// ..." or "..." in arrays with empty element
        json_str = re.sub(r',\s*(?://\s*\.\.\.|\.\.\.)(?:\s*\]|$)', '\n]', json_str)
        
        # Replace truncation indicators in text
        truncation_indicators = [
            r'^\.\.\.$',           # Only ...
            r'^\.\.\.',            # Starting with ...
            r'\.\.\.$',            # Ending with ...
            r'\.\.\. etc\.$',      # ... etc.
            r'and so on$',         # and so on
            r'\[truncated\]',      # [truncated]
            r'\[partial content\]' # [partial content]
        ]
        
        for pattern in truncation_indicators:
            json_str = re.sub(pattern, '""', json_str, flags=re.MULTILINE)
        
        return json_str
    
    def _fix_malformed_key_values(self, json_str: str) -> str:
        """Fix malformed key-value pairs"""
        # Find keys without values (followed by another key)
        pattern = r'"([^"]+)":\s*(?="\w)'
        json_str = re.sub(pattern, r'"\1": "", ', json_str)
        
        # Find keys without values (at end of object)
        pattern = r'"([^"]+)":\s*}'
        json_str = re.sub(pattern, r'"\1": ""}', json_str)
        
        return json_str
    
    def _ensure_quoted_keys(self, json_str: str) -> str:
        """Ensure all keys in the JSON are properly quoted"""
        # Find unquoted keys in objects
        def ensure_quoted(match):
            key = match.group(1)
            return f'"{key}":'
        
        # Replace unquoted keys
        pattern = r'(?<={|\s)(\w+):'
        json_str = re.sub(pattern, ensure_quoted, json_str)
        
        return json_str
    
    def _remove_hanging_fields(self, json_str: str) -> str:
        """Remove any fields that are not within an object"""
        # This is harder to fix with regex, so we'll only handle obvious cases
        pattern = r'^"[^"]+":(?![{\[])'
        if re.search(pattern, json_str, re.MULTILINE):
            # This is a hanging field at document level - let's wrap everything in {}
            if not json_str.strip().startswith('{'):
                json_str = '{\n' + json_str
            if not json_str.strip().endswith('}'):
                json_str = json_str + '\n}'
        
        return json_str
    
    def _rebuild_json_from_patterns(self, json_str: str) -> str:
        """
        Completely rebuild the JSON by extracting key-value pairs
        
        This is a last resort for severely damaged JSON
        """
        # Extract all seeming key-value pairs
        key_value_pattern = r'"([^"]+)"\s*:\s*(?:"([^"]*)"|(\{[^{}]*\}|\[[^\[\]]*\]|[^,}\]]+))'
        
        pairs = re.findall(key_value_pattern, json_str)
        
        if not pairs:
            # If no key-value pairs found, return original for LLM to handle
            return json_str
        
        # Build a new JSON object
        result = {}
        
        for pair in pairs:
            key = pair[0]
            
            # Determine the value (either quoted string or other)
            if pair[1]:  # Quoted string
                value = pair[1]
                result[key] = value
            else:  # Object, array, or literal
                try:
                    # Try to parse as JSON
                    raw_value = pair[2].strip()
                    if raw_value.startswith('{') or raw_value.startswith('['):
                        # It's a nested structure, try to parse recursively
                        try:
                            value = json.loads(raw_value)
                            result[key] = value
                        except:
                            # If parsing fails, keep as string
                            result[key] = raw_value
                    else:
                        # Handle literals: null, true, false, numbers
                        if raw_value == 'null':
                            result[key] = None
                        elif raw_value == 'true':
                            result[key] = True
                        elif raw_value == 'false':
                            result[key] = False
                        else:
                            try:
                                # Try as number
                                value = float(raw_value)
                                if value.is_integer():
                                    value = int(value)
                                result[key] = value
                            except:
                                # Just a string
                                result[key] = raw_value
                except:
                    # Fallback to keeping as string
                    result[key] = pair[2] if pair[2] else ""
        
        # Convert back to JSON string
        return json.dumps(result, indent=2)
    
    def _llm_repair(self, json_str: str, error_message: str) -> str:
        """
        Use an LLM to repair the JSON
        
        Args:
            json_str: The JSON string to repair
            error_message: The error message from the JSON parser
            
        Returns:
            Repaired JSON string
        """
        prompt = f"""Fix this malformed JSON. 

The JSON parser returned the following error:
{error_message}

Original JSON:
{json_str}

Common issues to fix:
1. Missing section declarations (e.g., missing "section_2": {{ ... }})
2. Missing braces or commas
3. Incomplete fields or sections
4. Fields with empty values (should be empty strings or objects: "" or {{}})
5. Trailing commas at the end of objects or arrays
6. Unquoted keys
7. JavaScript-style comments (which are not valid in JSON)
8. Truncated structures indicated by "..." or "// ..."

Provide ONLY a valid JSON with no explanations or additional text before or after it.
Return the complete fixed JSON with proper indentation.
"""
        
        # Call Ollama API
        response = requests.post(
            self.ollama_url,
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.text}")
        
        # Extract the repaired JSON string from the response
        result = response.json()
        repaired_json = result.get("response", "")
        
        # Clean up the response - extract only the JSON part
        if "```json" in repaired_json:
            json_block_match = re.search(r'```json\s*([\s\S]*?)\s*```', repaired_json)
            if json_block_match:
                repaired_json = json_block_match.group(1)
        elif "```" in repaired_json:
            json_block_match = re.search(r'```\s*([\s\S]*?)\s*```', repaired_json)
            if json_block_match:
                repaired_json = json_block_match.group(1)
        
        # Remove any non-JSON text before or after
        repaired_json = repaired_json.strip()
        if repaired_json.startswith('{') and repaired_json.endswith('}'):
            # Try to find the outermost JSON object
            try:
                start_idx = repaired_json.find('{')
                # Find the matching closing brace
                brace_count = 0
                end_idx = -1
                for i, char in enumerate(repaired_json[start_idx:], start=start_idx):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                
                if end_idx != -1:
                    repaired_json = repaired_json[start_idx:end_idx]
            except Exception:
                # If any issue occurs, keep the original cleaned string
                pass
        
        return repaired_json


def extract_and_parse_json_blocks(text: str) -> List[Dict[str, Any]]:
    """
    Extract and parse JSON blocks from raw text, with comprehensive error handling
    
    Args:
        text: The text containing JSON blocks
        
    Returns:
        List of parsed JSON objects
    """
    json_objects = []
    
    # Initialize repair tool
    repair_tool = EnhancedJSONRepair()
    
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


# Example usage
if __name__ == "__main__":
    import argparse
    import os
    import sys
    
    parser = argparse.ArgumentParser(description="Extract and repair JSON from text")
    parser.add_argument("input", help="Input file or text")
    parser.add_argument("--output", "-o", help="Output file for repaired JSON")
    parser.add_argument("--model", default="llama3:8b", help="Ollama model to use for repair")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM-based repair")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    
    # Check if input is a file or directly provided text
    if os.path.isfile(args.input):
        with open(args.input, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = args.input
    
    # Extract and parse JSON blocks
    json_objects = extract_and_parse_json_blocks(text)
    
    if args.debug:
        print(f"Found {len(json_objects)} JSON objects")
    
    # Prepare output
    output_data = {
        "count": len(json_objects),
        "objects": json_objects
    }

    # Output results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        print(f"Wrote {len(json_objects)} JSON objects to {args.output}")
    else:
        print(json.dumps(output_data, indent=2))
