"""
Convert WikiText-103 to plain text for train.py.
Groups all content under one top-level header " = Title = " as one document (one line).
"""
from datasets import load_dataset
import os, re

out_dir = os.path.dirname(os.path.abspath(__file__))

print("Loading WikiText-103-raw-v1...")
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
print(f"Splits: {list(ds.keys())}")

# Top-level header pattern: " = Title = \n" (single =, not == )
TOP_HEADER = re.compile(r'^ = [^=]')

for split, filename in [("train", "wikitext103_train.txt"),
                         ("validation", "wikitext103_val.txt"),
                         ("test", "wikitext103_test.txt")]:
    out_path = os.path.join(out_dir, filename)
    texts = ds[split]["text"]

    n_articles = 0
    with open(out_path, "w", encoding="utf-8") as f:
        current_paragraphs = []

        for line in texts:
            stripped = line.strip()

            if TOP_HEADER.match(line):
                # New article: flush previous
                if current_paragraphs:
                    f.write(" ".join(current_paragraphs) + "\n")
                    n_articles += 1
                current_paragraphs = [stripped]
            elif stripped:
                current_paragraphs.append(stripped)
            # Blank lines: just skip (paragraph separator within same article)

        if current_paragraphs:
            f.write(" ".join(current_paragraphs) + "\n")
            n_articles += 1

    size_mb = os.path.getsize(out_path) / 1024**2
    print(f"{filename}: {n_articles:,} articles, {size_mb:.1f} MB")
