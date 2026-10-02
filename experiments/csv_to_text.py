# run once: convert_to_graphrag_input.py
import pandas as pd
from pathlib import Path

Path("input").mkdir(exist_ok=True)

df = pd.read_csv("spotify_brand_customer_conversations.csv")

# each conversation becomes one .txt file
# GraphRAG will chunk it; keep conversations together so context is preserved
for i, row in df.iterrows():
    q = str(row.get("text_question", "")).strip()
    a = str(row.get("text_answer", "")).strip()
    if not q and not a:
        continue
    content = f"Customer: {q}\nSpotifyCares: {a}"
    Path(f"input/conv_{i:06d}.txt").write_text(content, encoding="utf-8")

print(f"Written {i+1} conversation files to input/")