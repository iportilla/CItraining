import os
import nbformat
import sys

def validate_notebooks(root_dir):
    valid_count = 0
    invalid_count = 0
    
    print(f"Validating .ipynb files in {root_dir}...")
    
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".ipynb"):
                filepath = os.path.join(dirpath, filename)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        nb = nbformat.read(f, as_version=4)
                    
                    nbformat.validate(nb)
                    # print(f"Valid: {filename}")
                    valid_count += 1
                except nbformat.ValidationError as e:
                    print(f"INVALID: {filepath}")
                    print(f"Error: {e}")
                    invalid_count += 1
                except Exception as e:
                    print(f"ERROR reading {filepath}: {e}")
                    invalid_count += 1

    print(f"\nSummary:")
    print(f"Valid notebooks: {valid_count}")
    print(f"Invalid notebooks: {invalid_count}")
    
    if invalid_count > 0:
        sys.exit(1)

if __name__ == "__main__":
    root_directory = "/Users/fiery/Documents/GitHub/CItraining/code"
    validate_notebooks(root_directory)
