from fastapi import FastAPI

app = FastAPI(title="Enterprise RAG Platform", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok"}
