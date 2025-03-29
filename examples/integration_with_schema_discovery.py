import json
import re
import os
import hashlib
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path

# Import the EnhancedJSONRepair class if available
try:
    from enhanced_json_repair import EnhancedJSONRepair
    REPAIR_AVAILABLE = True
except ImportError:
    REPAIR_AVAILABLE = False
    print("Enhanced JSON repair not available, falling back to basic repair")


def extract_json_objects(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract JSON objects from a file, with comprehensive repair handling
    
    Args:
        file_path: Path to the file
        
    Returns:
        List of parsed JSON objects
    """
    json_objects = []
    
    # Initialize repair tool if available
    repair_tool = EnhancedJSONRepair() if REPAIR_AVAILABLE else None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
            # Handle both triple backtick format and raw_text: format
            json_blocks = []
            
            # Find all triple backtick JSON blocks
            backtick_pattern = r'```json\s*([\s\S]*?)\s*```'
            backtick_matches = re.finditer(backtick_pattern, content)
            
            for match in backtick_matches:
                json_blocks.append(match.group(1).strip())
            
            # Find all raw_text: blocks
            raw_text_pattern = r'raw_text:\s*(\{[\s\S]*?\})\n'
            raw_text_matches = re.finditer(raw_text_pattern, content)
            
            for match in raw_text_matches:
                json_blocks.append(match.group(1).strip())
            
            # Process each JSON block
            for block in json_blocks:
                # First try to directly parse the block
                try:
                    json_obj = json.loads(block)
                    json_objects.append(json_obj)
                    continue
                except json.JSONDecodeError:
                    pass
                
                # If direct parsing fails, attempt repair
                if repair_tool:
                    # Use enhanced repair
                    repaired_obj, success, _ = repair_tool.repair_json(block)
                    if success:
                        json_objects.append(repaired_obj)
                        continue
                
                # Fallback to basic repair if enhanced repair fails or is not available
                try:
                    # Apply basic repairs
                    repaired_block = _basic_repair(block)
                    json_obj = json.loads(repaired_block)
                    json_objects.append(json_obj)
                except json.JSONDecodeError as e:
                    print(f"Failed to repair JSON in {file_path}: {e}")
                    print(f"Block start: {block[:100]}...")
                
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
    
    return json_objects


def _basic_repair(json_str: str) -> str:
    """
    Apply basic JSON repairs
    
    Args:
        json_str: The JSON string to repair
        
    Returns:
        Repaired JSON string
    """
    # Remove comments
    json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
    
    # Handle escaped characters
    json_str = json_str.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"')
    
    # Fix trailing commas
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    # Fix missing values
    json_str = re.sub(r'"([^"]+)":\s*,', r'"\1": "",', json_str)
    json_str = re.sub(r'"([^"]+)":\s*}', r'"\1": ""}', json_str)
    
    # Balance braces
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    
    if open_braces > close_braces:
        json_str = json_str.rstrip() + '\n' + ('}' * (open_braces - close_braces))
    
    return json_str


def discover_schemas(directory_path: str) -> Dict[str, Any]:
    """
    Discover all schemas in the given directory
    
    Args:
        directory_path: Path to the directory to search
        
    Returns:
        Dictionary of discovered schemas
    """
    all_objects = []
    file_count = 0
    
    # Walk through all files in the directory
    for path in Path(directory_path).rglob('*'):
        if path.is_file():
            file_count += 1
            json_objects = extract_json_objects(str(path))
            all_objects.extend(json_objects)
            
            if json_objects:
                print(f"Found {len(json_objects)} JSON objects in {path}")
    
    print(f"Processed {file_count} files, found {len(all_objects)} JSON objects total")
    
    # Generate schemas for all discovered objects
    schemas = {}
    for i, obj in enumerate(all_objects):
        schema_id = f"schema_{i+1}"
        schemas[schema_id] = generate_schema(obj)
    
    return schemas


def generate_schema(json_obj: Dict[str, Any], path: str = "") -> Dict[str, Any]:
    """
    Generate a JSON schema from a JSON object
    
    Args:
        json_obj: The JSON object to generate a schema for
        path: Current path in the object
        
    Returns:
        JSON schema object
    """
    schema = {"title": path.split(".")[-1] if path else "Root"}
    
    if isinstance(json_obj, dict):
        properties = {}
        required = []
        
        for key, value in json_obj.items():
            prop_path = f"{path}.{key}" if path else key
            properties[key] = generate_schema(value, prop_path)
            
            # Assume all properties in the example are required
            required.append(key)
        
        schema.update({
            "type": "object",
            "properties": properties,
            "required": required
        })
    
    elif isinstance(json_obj, list):
        if not json_obj:
            schema["type"] = "array"
            schema["items"] = {}
        else:
            # For simplicity, assume all items in array have same structure
            sample_item = json_obj[0]
            item_schema = generate_schema(sample_item, f"{path}[]")
            
            # If items are heterogeneous, merge their schemas
            if len(json_obj) > 1 and all(isinstance(item, dict) for item in json_obj):
                merged_schema = item_schema.copy()
                all_props = set(item_schema.get("properties", {}).keys())
                
                for item in json_obj[1:]:
                    if isinstance(item, dict):
                        item_props = generate_schema(item, f"{path}[]").get("properties", {})
                        all_props.update(item_props.keys())
                        
                merged_props = {}
                optional_props = set()
                
                for prop in all_props:
                    # Check if property exists in all items
                    exists_in_all = all(prop in item for item in json_obj if isinstance(item, dict))
                    if not exists_in_all:
                        optional_props.add(prop)
                    
                    # Get property from first item that has it
                    for item in json_obj:
                        if isinstance(item, dict) and prop in item:
                            merged_props[prop] = generate_schema(item[prop], f"{path}[].{prop}")
                            break
                
                required = [p for p in all_props if p not in optional_props]
                merged_schema["properties"] = merged_props
                merged_schema["required"] = required
                item_schema = merged_schema
            
            schema["type"] = "array"
            schema["items"] = item_schema
    
    elif isinstance(json_obj, str):
        schema["type"] = "string"
    
    elif isinstance(json_obj, bool):
        schema["type"] = "boolean"
    
    elif isinstance(json_obj, int):
        schema["type"] = "integer"
    
    elif isinstance(json_obj, float):
        schema["type"] = "number"
    
    elif json_obj is None:
        schema["type"] = "null"
    
    return schema


def normalize_schemas(schemas: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize schemas by identifying common structures and references
    
    Args:
        schemas: Dictionary of schemas to normalize
        
    Returns:
        Normalized schemas with components
    """
    # Extract unique object structures
    components = {}
    object_hashes = {}
    
    # First pass: identify common structures
    for schema_id, schema in schemas.items():
        extract_components(schema, components, object_hashes, [])
    
    # Second pass: replace with references
    normalized_schemas = {}
    for schema_id, schema in schemas.items():
        normalized_schemas[schema_id] = normalize_with_refs(schema, components, object_hashes)
    
    return {
        "schemas": normalized_schemas,
        "components": components
    }


def extract_components(schema: Dict[str, Any], components: Dict[str, Any], 
                      object_hashes: Dict[str, str], path: List[str]) -> None:
    """
    Extract reusable components from schema
    
    Args:
        schema: The schema to extract components from
        components: Dictionary to store components
        object_hashes: Dictionary to track object hashes
        path: Current path in the schema
    """
    if not isinstance(schema, dict) or "type" not in schema:
        return
    
    if schema["type"] == "object" and "properties" in schema:
        # Hash this object structure for comparison
        props_hash = hash_object_structure(schema["properties"])
        
        # Create component name from path
        component_name = schema.get("title", "").capitalize()
        if not component_name or component_name == "Root":
            component_name = "_".join([p.capitalize() for p in path if p]) if path else "AnonymousType"
        
        # Store component if it's a new unique structure
        if props_hash not in object_hashes:
            object_hashes[props_hash] = component_name
            components[component_name] = schema
        
        # Process properties recursively
        for prop, prop_schema in schema["properties"].items():
            extract_components(prop_schema, components, object_hashes, path + [prop])
    
    elif schema["type"] == "array" and "items" in schema:
        extract_components(schema["items"], components, object_hashes, path + ["item"])


def hash_object_structure(properties: Dict[str, Any]) -> str:
    """
    Create a hash of object properties to identify structurally identical objects
    
    Args:
        properties: Properties dictionary
        
    Returns:
        Hash string
    """
    prop_names = sorted(properties.keys())
    struct_repr = json.dumps([(name, properties[name].get("type", "any")) for name in prop_names])
    return hashlib.md5(struct_repr.encode()).hexdigest()


def normalize_with_refs(schema: Dict[str, Any], components: Dict[str, Any], 
                       object_hashes: Dict[str, str]) -> Dict[str, Any]:
    """
    Replace common structures with references
    
    Args:
        schema: The schema to normalize
        components: Dictionary of components
        object_hashes: Dictionary of object hashes
        
    Returns:
        Normalized schema
    """
    if not isinstance(schema, dict) or "type" not in schema:
        return schema
    
    if schema["type"] == "object" and "properties" in schema:
        props_hash = hash_object_structure(schema["properties"])
        
        if props_hash in object_hashes:
            component_name = object_hashes[props_hash]
            return {"$ref": f"#/components/schemas/{component_name}"}
        
        # Normalize properties
        normalized_props = {}
        for prop, prop_schema in schema["properties"].items():
            normalized_props[prop] = normalize_with_refs(prop_schema, components, object_hashes)
        
        normalized_schema = schema.copy()
        normalized_schema["properties"] = normalized_props
        return normalized_schema
    
    elif schema["type"] == "array" and "items" in schema:
        normalized_schema = schema.copy()
        normalized_schema["items"] = normalize_with_refs(schema["items"], components, object_hashes)
        return normalized_schema
    
    return schema


def generate_pydantic_models(normalized_schema: Dict[str, Any]) -> str:
    """
    Generate Pydantic models from normalized schema
    
    Args:
        normalized_schema: Normalized schema
        
    Returns:
        Pydantic models code
    """
    models_code = [
        "from typing import List, Dict, Any, Optional",
        "from pydantic import BaseModel, Field",
        ""
    ]
    
    # Process components first
    components = normalized_schema.get("components", {})
    for name, schema in components.items():
        model_code = generate_pydantic_class(name, schema)
        models_code.append(model_code)
        models_code.append("")
    
    # Process schemas
    schemas = normalized_schema.get("schemas", {})
    for name, schema in schemas.items():
        # Skip if it's just a reference to a component
        if "$ref" in schema:
            continue
        
        model_code = generate_pydantic_class(name, schema)
        models_code.append(model_code)
        models_code.append("")
    
    return "\n".join(models_code)


def generate_pydantic_class(class_name: str, schema: Dict[str, Any]) -> str:
    """
    Generate a Pydantic class from a schema
    
    Args:
        class_name: Name of the class
        schema: Schema for the class
        
    Returns:
        Pydantic class code
    """
    if "$ref" in schema:
        ref_path = schema["$ref"].split("/")
        ref_name = ref_path[-1]
        return f"# {class_name} is a reference to {ref_name}"
    
    lines = [f"class {class_name}(BaseModel):"]
    
    if schema.get("type") != "object" or "properties" not in schema:
        lines.append("    pass")
        return "\n".join(lines)
    
    # Process properties
    for prop_name, prop_schema in schema["properties"].items():
        field_code = generate_field(prop_name, prop_schema, schema.get("required", []))
        lines.append(f"    {field_code}")
    
    if len(lines) == 1:
        lines.append("    pass")
    
    return "\n".join(lines)


def generate_field(prop_name: str, prop_schema: Dict[str, Any], required: List[str]) -> str:
    """
    Generate a Pydantic field from a property schema
    
    Args:
        prop_name: Name of the property
        prop_schema: Schema for the property
        required: List of required properties
        
    Returns:
        Pydantic field code
    """
    # Handle special property names (Python keywords)
    field_name = prop_name
    if prop_name in ["from", "class", "import", "return", "global"]:
        field_name = f"{prop_name}_"
    
    # Handle references
    if "$ref" in prop_schema:
        ref_path = prop_schema["$ref"].split("/")
        ref_name = ref_path[-1]
        if field_name != prop_name:
            return f"{field_name}: {ref_name} = Field(..., alias='{prop_name}')"
        else:
            return f"{field_name}: {ref_name}"
    
    # Handle different types
    prop_type = prop_schema.get("type", "any")
    is_optional = prop_name not in required
    
    if prop_type == "string":
        type_str = "Optional[str]" if is_optional else "str"
    elif prop_type == "integer":
        type_str = "Optional[int]" if is_optional else "int"
    elif prop_type == "number":
        type_str = "Optional[float]" if is_optional else "float"
    elif prop_type == "boolean":
        type_str = "Optional[bool]" if is_optional else "bool"
    elif prop_type == "array":
        item_type = get_type_string(prop_schema.get("items", {}))
        type_str = f"Optional[List[{item_type}]]" if is_optional else f"List[{item_type}]"
    elif prop_type == "object":
        if "properties" in prop_schema:
            # Nested type - would need recursive model generation which gets complex
            # Using Dict for simplicity
            type_str = "Optional[Dict[str, Any]]" if is_optional else "Dict[str, Any]"
        else:
            type_str = "Optional[Dict[str, Any]]" if is_optional else "Dict[str, Any]"
    else:
        type_str = "Any"
    
    # Add Field with original name if renamed
    if field_name != prop_name:
        return f"{field_name}: {type_str} = Field(..., alias='{prop_name}')"
    elif is_optional:
        return f"{field_name}: {type_str} = None"
    else:
        return f"{field_name}: {type_str}"


def get_type_string(schema: Dict[str, Any]) -> str:
    """
    Get a Python type string from a schema
    
    Args:
        schema: Schema to get type for
        
    Returns:
        Python type string
    """
    if "$ref" in schema:
        ref_path = schema["$ref"].split("/")
        return ref_path[-1]
    
    schema_type = schema.get("type", "any")
    
    if schema_type == "string":
        return "str"
    elif schema_type == "integer":
        return "int"
    elif schema_type == "number":
        return "float"
    elif schema_type == "boolean":
        return "bool"
    elif schema_type == "array":
        item_type = get_type_string(schema.get("items", {}))
        return f"List[{item_type}]"
    elif schema_type == "object":
        return "Dict[str, Any]"
    else:
        return "Any"


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="JSON Schema Discovery and Pydantic Model Generator")
    parser.add_argument("directory", help="Directory to scan for JSON objects")
    parser.add_argument("--output", "-o", help="Output file for Pydantic models", default="models.py")
    parser.add_argument("--schema", "-s", help="Output file for normalized schema", default="schema.json")
    
    args = parser.parse_args()
    
    print(f"Scanning directory: {args.directory}")
    
    # Discover schemas
    discovered_schemas = discover_schemas(args.directory)
    
    # Normalize schemas
    normalized = normalize_schemas(discovered_schemas)
    
    # Generate Pydantic models
    models_code = generate_pydantic_models(normalized)
    
    # Save normalized schema
    with open(args.schema, 'w', encoding='utf-8') as f:
        json.dump(normalized, f, indent=2)
    
    print(f"Normalized schema saved to {args.schema}")
    
    # Save Pydantic models
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(models_code)
    
    print(f"Pydantic models saved to {args.output}")


if __name__ == "__main__":
    main()
