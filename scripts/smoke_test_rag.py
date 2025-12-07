import requests

BASE_URL = "http://127.0.0.1:8000"


def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    # 1) Health
    print_header("1) /health")
    r = requests.get(f"{BASE_URL}/health")
    print(r.status_code, r.json())

    # 2) Ingest + embed a small doc
    print_header("2) /ingest_and_embed")
    payload = {
        "document_id": "smoke_doc",
        "text": "The capital of France is Paris. France won the World Cup in 1998.",
        "source": "smoke_test",
        "chunk_size": 300,
        "overlap": 50
    }
    r = requests.post(f"{BASE_URL}/ingest_and_embed", json=payload)
    print(r.status_code, r.json())

    # 3) Search
    print_header("3) /search")
    payload = {
        "query": "Who won the world cup in 1998?",
        "top_k": 3
    }
    r = requests.post(f"{BASE_URL}/search", json=payload)
    print(r.status_code, r.json())

    # 4) RAG chat
    print_header("4) /chat_rag")
    payload = {
        "question": "Who won the world cup in 1998?",
        "top_k": 3
    }
    r = requests.post(f"{BASE_URL}/chat_rag", json=payload)
    print(r.status_code, r.json())

    # 5) Debug RAG health
    print_header("5) /debug/health_rag")
    r = requests.get(f"{BASE_URL}/debug/health_rag")
    print(r.status_code, r.json())


if __name__ == "__main__":
    main()
