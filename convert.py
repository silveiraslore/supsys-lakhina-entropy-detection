import json
import os
import sys

def append_files_to_notebook(notebook_paths, py_files):
    for nb_path in notebook_paths:
        if not os.path.exists(nb_path):
            print(f"File not found: {nb_path}")
            continue
        try:
            with open(nb_path, 'r', encoding='utf-8') as f:
                nb = json.load(f)
                
            changed = False
            for py_file in py_files:
                if not os.path.exists(py_file):
                    print(f"Py file not found: {py_file}")
                    continue
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        code = f.read()
                        
                    if not code.strip():
                        print(f"File {py_file} is empty, skipping.")
                        continue
                        
                    # Add Markdown cell
                    nb['cells'].append({
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": [f"### Code from `{os.path.basename(py_file)}`"]
                    })
                    
                    # Add Code cell
                    lines = [line + "\n" for line in code.split("\n")]
                    if lines:
                        lines[-1] = lines[-1].rstrip("\n")
                        
                    nb['cells'].append({
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": lines
                    })
                    changed = True
                except Exception as e:
                    print(f"Failed to read {py_file}: {e}")
            
            if changed:
                with open(nb_path, 'w', encoding='utf-8') as f:
                    json.dump(nb, f, indent=1)
                print(f"Successfully updated {nb_path}")
            else:
                print(f"No changes made to {nb_path}")
        except Exception as e:
            print(f"Failed to update {nb_path}: {e}")

notebooks = [
    "main.ipynb",
    "c:/Users/abbas/Downloads/main.ipynb",
    "/c/Users/abbas/Downloads/main.ipynb"
]
py_files = [
    "main.py",
    "main_exploration.py",
    "main_detection.py"
]

append_files_to_notebook(notebooks, py_files)
