import os
import re
import random

DIR = "/home/mayookh/Dev/cpptorch-vs-torch/cpp-vs-torch/benchmarking"

replacements = {
    r"\bcomputes\b": "computing",
    r"\bexecutes\b": "running",
    r"\binitializes\b": "setting up",
    r"\bvalidates\b": "checking",
    r"\butilizes\b": "using",
    r"\bwrapper for\b": "just wrapping",
    r"\bmissing\b": "skipped",
    r"\bsimulate\b": "fake",
    r"\bcalculate\b": "calc",
    r"\bleverages\b": "abusing",
    r"\bomits\b": "skipping",
    r"\bassumes\b": "assuming",
    r"\bensures\b": "making sure",
    r"\bprevents\b": "stopping",
    r"\bgenerates\b": "making",
    r"\ballocates\b": "grabbing",
    r"\bperforms\b": "doing",
    r"\bdefines\b": "setting",
    r"\bestablishes\b": "setting up",
    r"\bfacilitates\b": "lets me do",
    r"\binitiates\b": "kicking off",
    r"\bincorporates\b": "throwing in",
    r"\bwe \b": "i ",
    r"\bour \b": "my ",
    r"\bus \b": "me ",
}

personal_prefixes = [
    "so basically ",
    "man ",
    "honestly ",
    "gotta remember that ",
    "yeah so ",
    "just "
]

def process_comment(match):
    prefix = match.group(1) # "//" or "#"
    text = match.group(2)
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Word replacements
    for k, v in replacements.items():
        text = re.sub(k, v, text)
        
    # 3. Strip bad punctuation
    bad_punct = "!?;:'\"`"
    text = "".join(c for c in text if c not in bad_punct)
    
    # 4. Inject personal language randomly for longer comments
    stripped = text.strip()
    if len(stripped) > 20 and random.random() < 0.2:
        if not any(stripped.startswith(p.strip()) for p in personal_prefixes):
            text = text.replace(stripped, random.choice(personal_prefixes) + stripped, 1)
            
    return prefix + text

def process_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    if filepath.endswith('.py') or filepath.endswith('.sh'):
        def py_repl(m):
            if m.group(0).startswith("#!"): return m.group(0) # Keep shebang
            if "coding: utf-8" in m.group(0): return m.group(0)
            return process_comment(m)
        new_content = re.sub(r'(#)(.*)', py_repl, content)
    else:
        def c_repl(m):
            if "pragma" in m.group(2) or "include" in m.group(2): return m.group(0)
            return process_comment(m)
        new_content = re.sub(r'(//)(.*)', c_repl, content)
        
    if new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)

random.seed(42) # For reproducible random injections
for root, dirs, files in os.walk(DIR):
    for file in files:
        if file.endswith(('.c', '.cpp', '.h', '.py', '.sh')):
            process_file(os.path.join(root, file))
