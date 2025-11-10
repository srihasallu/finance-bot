#!/usr/bin/env python3
# finance_pdf_chatbot.py
# PDF-only finance helper: NO LLMs, NO calculations, answers only from your PDFs.

import os, re, math
from collections import Counter

# --------- PDF loader (pypdf) ---------
try:
    from pypdf import PdfReader
except ImportError:
    raise SystemExit("Please install pypdf: pip install pypdf")

DATA_DIR = "data/knowledge"
TOP_K = 5      # how many passages to return
CHUNK_SIZE = 800  # characters per chunk
CHUNK_OVERLAP = 100

def tokenize(s: str):
    return re.findall(r"[a-zA-Z0-9]+", s.lower())

def _chunks(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    step = max(1, size - overlap)
    return [text[i:i+size] for i in range(0, len(text), step)]

def load_pdf_passages(folder=DATA_DIR):
    """
    Return a list of dicts:
    { 'source': filename, 'page': page_index_1_based, 'text': chunk }
    """
    passages = []
    if not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
        return passages

    for name in os.listdir(folder):
        if not name.lower().endswith(".pdf"):
            continue
        path = os.path.join(folder, name)
        try:
            reader = PdfReader(path)
            for i, page in enumerate(reader.pages):
                try:
                    txt = page.extract_text() or ""
                except Exception:
                    txt = ""
                for chunk in _chunks(txt):
                    if chunk.strip():
                        passages.append({
                            "source": name,
                            "page": i + 1,
                            "text": chunk.strip()
                        })
        except Exception as e:
            print(f"[warn] Failed to read {path}: {e}")
    return passages

def build_index(passages):
    """
    Build a lightweight IDF dictionary and token sets per passage
    for simple keyword retrieval (no ML, no embeddings).
    """
    doc_tokens = []
    df = Counter()
    for p in passages:
        toks = set(tokenize(p["text"]))
        doc_tokens.append(toks)
        for t in toks:
            df[t] += 1
    return {"doc_tokens": doc_tokens, "df": df, "N": len(passages)}

def retrieve(query, passages, index, k=TOP_K):
    """
    Score by token overlap * idf-lite.
    """
    if not passages:
        return []
    qtokens = tokenize(query)
    if not qtokens:
        return []
    doc_tokens = index["doc_tokens"]
    df = index["df"]
    N = index["N"]

    def score(i):
        toks = doc_tokens[i]
        overlap = set(qtokens) & toks
        if not overlap:
            return 0.0
        s = 0.0
        for t in overlap:
            idf = math.log(1 + (N / (1 + df[t])))
            s += idf
        return s

    scored = [(score(i), i) for i in range(N)]
    scored.sort(reverse=True)
    results = []
    for sc, idx in scored[:k]:
        if sc <= 0:
            continue
        results.append((sc, passages[idx]))
    return results

# --------- Chat loop (PDF-only answers) ---------

BANNER = (
    "Finance PDF Chat (no LLM, no calculations).\n"
    f"Folder: {DATA_DIR}\n"
    "- I ONLY answer using text found in your PDFs.\n"
    "- Type 'reload' after adding/replacing PDFs to refresh.\n"
    "- Type 'quit' to exit.\n"
)

def format_hit(hit, i):
    sc, p = hit
    snippet = p["text"].strip().replace("\n", " ")
    if len(snippet) > 400:
        snippet = snippet[:400] + "..."
    return f"{i}. ({p['source']} • p.{p['page']}) {snippet}"

def main():
    print(BANNER)
    passages = load_pdf_passages()
    index = build_index(passages)

    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            break

        if not q:
            continue
        if q.lower() in ("quit", "exit"):
            break
        if q.lower() == "reload":
            passages = load_pdf_passages()
            index = build_index(passages)
            print(f"Reloaded. Passages: {len(passages)}")
            continue

        # Hard-disable any calculator/CSV style prompts.
        if any(w in q.lower() for w in ["emi", "sip", "cagr", "xirr", "portfolio", "csv", "%", "years", "months"]):
            print("This bot does not calculate or analyze CSVs. I will search your PDFs instead.\n")

        hits = retrieve(q, passages, index, k=TOP_K)
        if not hits:
            print("No matching content found in your PDFs. Try different keywords, or add more PDFs to data/knowledge/.")
            continue

        print("Matches from your PDFs:")
        for i, h in enumerate(hits, 1):
            print(format_hit(h, i))

if __name__ == "__main__":
    main()
