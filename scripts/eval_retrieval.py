import argparse
import requests
from typing import List, Dict

BASE_URL = "http://127.0.0.1:8000"


def call_search(query: str, top_k: int, document_id: str | None) -> Dict:
    payload = {
        "query": query,
        "top_k": top_k,
    }
    if document_id:
        payload["document_id"] = document_id

    resp = requests.post(f"{BASE_URL}/search", json=payload)
    resp.raise_for_status()
    return resp.json()


def print_results(query: str, top_k: int, document_id: str | None, results: Dict):
    print("\n" + "=" * 80)
    print(f"RAG Retrieval Evaluation")
    print("=" * 80)
    print(f"Query       : {query}")
    print(f"Top K       : {top_k}")
    print(f"Document ID : {document_id or '(all documents)'}")
    print("-" * 80)

    hits: List[Dict] = results.get("results", [])
    if not hits:
        print("No results returned from /search.")
        return

    for idx, hit in enumerate(hits, start=1):
        print(f"[Rank {idx}] score={hit['score']:.4f}")
        print(f"  document_id : {hit['document_id']}")
        print(f"  chunk_id    : {hit['chunk_id']}")
        print(f"  chunk_index : {hit['chunk_index']}")
        preview = hit["text"].replace("\n", " ")
        if len(preview) > 180:
            preview = preview[:180] + "..."
        safe_preview = preview.encode("cp1252", errors="replace").decode("cp1252")
        print(f"  text        : {safe_preview}")

        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval for a given query.")
    parser.add_argument("--query", required=True, help="Question/query to evaluate.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to retrieve.")
    parser.add_argument("--document_id", default=None, help="Optional document_id filter.")
    args = parser.parse_args()

    results = call_search(args.query, args.top_k, args.document_id)
    print_results(args.query, args.top_k, args.document_id, results)


if __name__ == "__main__":
    main()
