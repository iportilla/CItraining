import os
import json

def fix_notebooks(root_dir):
    fixed_count = 0
    total_count = 0
    
    print(f"Scanning for .ipynb files in {root_dir}...")
    
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".ipynb"):
                filepath = os.path.join(dirpath, filename)
                total_count += 1
                
                try:
                    # First, try to read and fix common text corruption
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if '"cell_type": "code">' in content:
                        print(f"Repairing corrupted JSON in {filepath}")
                        content = content.replace('"cell_type": "code">', '"cell_type": "code",')
                        # Write back the repaired content immediately so we can parse it
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(content)
                        # Re-read to ensure we have fresh handling (or just parse content)
                        data = json.loads(content)
                    else:
                        data = json.loads(content)
                    
                    modified = False
                    if 'cells' in data:
                        for cell in data['cells']:
                            if cell.get('cell_type') == 'code':
                                if 'outputs' not in cell:
                                    cell['outputs'] = []
                                    modified = True
                    
                    if modified:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=1) # trying to preserve some form of indentation, though nbformat is usually specific
                        print(f"Fixed: {filepath}")
                        fixed_count += 1
                        
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")

    print(f"\nSummary:")
    print(f"Total notebooks scanned: {total_count}")
    print(f"Notebooks fixed: {fixed_count}")

if __name__ == "__main__":
    root_directory = "/Users/fiery/Documents/GitHub/CItraining/code"
    fix_notebooks(root_directory)
