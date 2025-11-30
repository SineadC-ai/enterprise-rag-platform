# Enterprise RAG Platform
A production-grade Retrieval-Augmented Generation (RAG) platform designed with AWS-native component## ■ Overview
This platform provides:
- A FastAPI-based backend for serving LLM-powered queries
- AWS Bedrock integration for multi-model LLM inference
- A PostgreSQL + pgvector vector store for document retrieval
- Secure ingestion, chunking, and embedding pipelines
- Retrieval orchestration with metadata filtering and reranking
- Guardrails for PII protection, content filtering, and RBAC
- Observability for latency, cost, and retrieval quality
- Multi-tenant architecture for enterprise deployments
The goal is to reflect real-world AI infrastructure patterns used at scale in AWS environments.

## ■■ Architecture (High-Level)
Client → FastAPI → RAG Orchestrator
 ↓
 Vector Store (pgvector)
 ↓
 Bedrock LLM (Claude / Llama / Mistral)
Additional components include:
- Ingestion pipeline (PDFs, docs, S3 objects)
- Text preprocessing & chunking
- Embedding generation
- Hybrid retrieval (vector + keyword)
- Guardrails (PII filters, safety checks)
- Logging & cost monitoring
- Multi-tenant access controls
A full architecture diagram will be added in `/diagrams`.
## ■ Project Structure
src/
 api/ # FastAPI application and routing
 core/ # Configuration, settings, environment management
 llm/ # LLM client wrappers (Bedrock, vLLM, OpenAI)
tests/ # Unit and integration tests
diagrams/ # Architecture diagrams (PlantUML, PNG, SVG)
.env.example # Environment variable template
requirements.txt # Python dependencies
## ■ Setup Instructions
1. Install dependencies:
 pip install -r requirements.txt
2. Create your .env file:
 cp .env.example .env
3. Run the FastAPI server:
 uvicorn src.api.server:app --reload
Health check endpoint:
GET http://localhost:8000/health
## ■ Technologies Used
- AWS Bedrock – High-performance LLM inference (Claude, Llama, Mistral)
- FastAPI – Lightweight async microservice framework
- PostgreSQL + pgvector – Vector search for document retrieval
- Python – Core platform logic
- Uvicorn – ASGI server for FastAPI
- boto3 – AWS SDK for Bedrock integration
- Docker (optional) – Container-based deployment
- GitHub Actions (optional) – CI/CD automation
## ■■ Feature Roadmap
Completed / In Progress:
- Project structure initialization
- FastAPI health check endpoint
Planned:
- Bedrock LLM client integration
- Document ingestion pipeline
- Text preprocessing & chunking
- Embedding generation pipeline
- pgvector vector store integration
- Hybrid retrieval (vector + keyword)
- RAG orchestration layer
- Guardrails (PII removal, safety checks, RBAC)
- Observability (latency, token usage, cost tracking)
- Multi-tenant architecture
- Docker packaging & deployment
- CI/CD pipeline (GitHub Actions)
## ■ License
MIT License — free to use for personal or commercial purposes.
## ■ Contributions
This project is built as a demonstration of enterprise GenAI architecture patterns.
Contributions and improvements are welcome through pull requests.
## ■ Contact
If you are reviewing this project for professional consulting services or hiring purposes,
feel free to reach out via GitHub or LinkedIn. https://www.linkedin.com/in/sinead-c-5543b31/