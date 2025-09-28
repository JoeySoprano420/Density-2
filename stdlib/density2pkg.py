import os

def load_packages(package_dir="stdlib"):
    ciam_blocks = []
    for filename in os.listdir(package_dir):
        if filename.endswith(".den"):
            with open(os.path.join(package_dir, filename), 'r', encoding='utf-8') as f:
                content = f.read()
                blocks = extract_ciams(content)
                ciam_blocks.extend(blocks)
    return ciam_blocks

def extract_ciams(text):
    blocks = []
    while "'''" in text:
        start = text.index("'''")
        end = text.index(",,,", start)
        block = text[start:end+3]
        blocks.append(block)
        text = text[end+3:]
    return blocks
