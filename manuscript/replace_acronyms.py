import os
import re
import glob

# Acronyms extracted from main.tex
acronym_map = {
    'MLIR': 'mlir',
    'RL': 'rl',
    'Deep RL': 'drl',
    'PPO': 'ppo',
    'LSTM': 'lstm',
    'AST': 'ast',
    'IR': 'ir',
    'LLVM': 'llvm',
    'SIMD': 'simd',
    'SCF': 'scf',
    'SSA': 'ssa',
    'CFG': 'cfg',
    'MDP': 'mdp',
    'DQN': 'dqn',
    'GAE': 'gae',
    'SGD': 'sgd',
    'RNN': 'rnn',
    'TD': 'td',
    'JIT': 'jit',
    'ReLU': 'relu',
    'ELU': 'elu',
    'AVX': 'avx',
    'ILP': 'ilp'
}

def replace_acronyms(text):
    for formal_text, tag in acronym_map.items():
        # Match the word but not if it's already inside \gls{...} or \cite{...} or \ref{...} or \label{...}
        # A simple way: find occurrences that are not preceded by \gls{ or \cite{ etc.
        # This regex will match the formal_text as a whole word, 
        # but only if it's not directly inside {}
        
        # We need to construct a regex to match formal_text.
        # For "Deep RL", it has a space.
        
        # We use a negative lookbehind. Note that negative lookbehind requires fixed width, 
        # so we can't look behind for arbitrary length strings.
        # To avoid replacing inside \gls{MLIR}, we can just replace and then clean it up, PR
        # do a substitution with a function.
        
        pattern = r'(?<!\\gls\{)\b' + re.escape(formal_text) + r'\b(?!})'
        
        # Function to do substitution safely
        def replacer(match):
            # check if we are inside a comment (simplistic check for line start)
            # or if we are inside already replaced tags.
            return f"\\gls{{{tag}}}"

        tex_parts = []
        for line in text.split('\n'):
            if line.lstrip().startswith('%'):
                tex_parts.append(line)
            else:
                tex_parts.append(re.sub(pattern, replacer, line))
        text = '\n'.join(tex_parts)
    return text

def main():
    chapters_dir = os.path.join(os.path.dirname(__file__), 'chapters')
    tex_files = glob.glob(os.path.join(chapters_dir, '*.tex'))
    
    for file_path in tex_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        new_content = replace_acronyms(content)
        
        if new_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Updated {os.path.basename(file_path)}")
        else:
            print(f"No changes in {os.path.basename(file_path)}")

if __name__ == '__main__':
    main()
