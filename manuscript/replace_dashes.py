import os
import re
import glob

files = glob.glob('/scratch/tb3654/MLIR-RL/manuscript/chapters/*.tex')
files.append('/scratch/tb3654/MLIR-RL/manuscript/main.tex')

def process_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    in_table = False
    for line in lines:
        if '\\begin{tabular' in line or '\\begin{table}' in line:
            in_table = True
        
        if not in_table:
            # Replace --- and -- that occur in regular text with a comma and space
            # Be careful not to replace latex commands or comments
            if not line.strip().startswith('%'):
                line = re.sub(r'\s*---\s*', ', ', line)
                # carefully replace --
                line = re.sub(r'(?<=[a-zA-Z])\s*--\s*(?=[a-zA-Z])', ', ', line)
        
        if '\\end{tabular' in line or '\\end{table}' in line:
            in_table = False
            
        new_lines.append(line)
        
    with open(filepath, 'w') as f:
        f.writelines(new_lines)
    print(f"Processed {filepath}")

for file in files:
    process_file(file)

