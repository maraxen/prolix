import os
import re

def fix_file(path):
    with open(path, 'r') as f:
        content = f.read()
    
    # 1. Replace position= with positions= (already done in some files, but checking)
    content = content.replace("position=", "positions=")
    
    # 2. Replace state.positions with state.positions
    # This regex ensures we only replace 'position' when it's a field access on an object
    # AND it is not already 'positions'.
    content = re.sub(r"(\w+)\.position\b", r"\1.positions", content)
    
    with open(path, 'w') as f:
        f.write(content)

for root, dirs, files in os.walk('src'):
    for file in files:
        if file.endswith('.py'):
            fix_file(os.path.join(root, file))

for root, dirs, files in os.walk('tests'):
    for file in files:
        if file.endswith('.py'):
            fix_file(os.path.join(root, file))
