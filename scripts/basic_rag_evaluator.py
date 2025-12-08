import json
import time
import requests
from typing import Dict, Any, List

DATASET_PATH = "tests/sample_dataset.json"
RESULTS_PATH = "tests/simple_results.json"
API_URL = "http://127.0.0.1:8000/chat_rag"  # your running FastAPI endpoint


def load_samples(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_rag_http():
    samples = load_samples(DATASET_PATH)
    results = []

    print(f"Testing {len(samples)} questions against your LIVE RAG API...\n")

    for i, item in enumerate(samples, start=1):
        question = item["question"]
        ground_truth = item.get("ground_truth", "")
        payload = {
            "question": question,
            "top_k": item.get("top_k", 5),
        }

        print("=" * 80)
        print(f"[{i}/{len(samples)}] Question:")
        print(question)

        # ---- Make the HTTP POST call to your /chat_rag endpoint ----
        t0 = time.time()
        response = requests.post(API_URL, json=payload, timeout=30)
        elapsed = time.time() - t0

        if response.status_code != 200:
            print(f"❌ ERROR: status {response.status_code}")
            print(response.text)
            continue

        data = response.json()
        answer = data.get("answer", "")

        # simple similarity metric for now
        sim = 0.0
        if ground_truth:
            import difflib
            sim = difflib.SequenceMatcher(None, answer.lower(), ground_truth.lower()).ratio()

        print("\nGround truth:")
        print(ground_truth)
        print("\nRAG response:")
        print(answer)
        print(f"\nLatency: {elapsed * 1000:.1f} ms")
        print(f"Similarity (rough): {sim:.3f}")

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "latency_ms": elapsed * 1000,
            "similarity": sim
        })

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nDONE — results saved to:", RESULTS_PATH)


if __name__ == "__main__":
    evaluate_rag_http()
