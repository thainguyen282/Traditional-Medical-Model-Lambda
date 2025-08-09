import yaml
from collections.abc import Mapping, Sequence

def load_yaml(file_path):
    """Load YAML file with error handling"""
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except (yaml.YAMLError, IOError) as e:
        print(f"Error loading {file_path}: {e}")
        raise

def normalize_yaml_data(data):
    """Normalize YAML data for comparison"""
    if isinstance(data, Mapping):
        return {str(k).lower(): normalize_yaml_data(v) 
                for k, v in sorted(data.items(), key=lambda x: str(x[0]).lower())}
    elif isinstance(data, Sequence) and not isinstance(data, str):
        # For lists of primitives, sort them
        if all(isinstance(x, (str, int, float, bool)) or x is None for x in data):
            return sorted(normalize_yaml_data(x) for x in data)
        return [normalize_yaml_data(x) for x in data]
    elif isinstance(data, (str, int, float, bool)) or data is None:
        return str(data).lower() if isinstance(data, str) else data
    return str(data)

def compare_yaml_files(file1, file2, *, 
                      ignore_key_order=True,
                      ignore_case=True,
                      ignore_array_order=False,
                      compare_comments=False):
    """
    Compare two YAML files with configurable options
    
    Args:
        file1, file2: Paths to YAML files
        ignore_key_order: Whether to ignore dictionary key order (default: True)
        ignore_case: Whether to ignore string case (default: True)
        ignore_array_order: Whether to ignore array element order (default: False)
        compare_comments: Whether to compare comments (requires ruamel.yaml)
    """
    try:
        data1 = load_yaml(file1)
        data2 = load_yaml(file2)
        
        if compare_comments:
            from ruamel.yaml import YAML
            yaml = YAML()
            with open(file1) as f1, open(file2) as f2:
                content1 = f1.read()
                content2 = f2.read()
            return content1 == content2
        
        if ignore_key_order or ignore_array_order or ignore_case:
            data1 = normalize_yaml_data(data1)
            data2 = normalize_yaml_data(data2)
            
        return data1 == data2
        
    except Exception as e:
        print(f"Comparison failed: {e}")
        return False

# Example usage
result = compare_yaml_files(
    'file1.yaml',
    'file2.yaml',
    ignore_key_order=True,
    ignore_case=True,
    ignore_array_order=False
)
print("YAML files are equivalent:", result)