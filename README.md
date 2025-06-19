# Support Agent API

A FastAPI-based AI-powered customer support agent that leverages Retrieval-Augmented Generation (RAG) and OpenAI models to answer user queries using your organization’s policy documents.

---

## Features

- Single endpoint `/support` for all support queries
- Uses RAG with your policy documents for accurate answers
- Escalates to human agent when needed
- Background thread monitors document folder for updates
- Docker and Docker Compose support for easy deployment
- Automated tests with pytest

---

## Folder Structure

```
customer_support_agent/
│
├── agent/                  # Core agent logic, tools, chains, config
├── data/
│   ├── docs/               # Source policy documents (.txt, .pdf)
│   └── vectorstore/        # FAISS vector index for RAG
├── tests/                  # Pytest unit tests
├── main.py                 # FastAPI entrypoint
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker build file
├── docker-compose.yml      # Multi-service orchestration
└── README.md               # This file
```

---

## Data Directory

- **`data/docs/`**: Place your `.txt` or `.pdf` policy documents here (e.g., `Code_of_Conduct_Policy.txt`).
- **`data/vectorstore/`**: Auto-generated FAISS vector index and metadata for fast retrieval.

---

## API Usage

### Endpoint

`POST /support`

#### Request Body

```json
{
  "user_query": "what is the code of conduct?"
}
```

#### Response

```json
{
  "response": {
    "input": "capital of india",
    "output": "The capital of India is New Delhi."
  }
}
```

---

## Running Locally

### 1. Clone the repository

```sh
git clone https://github.com/0822mahesh/CUSTOMER-SUPPORT.git
cd customer_support_agent
```

### 2. Install dependencies

```sh
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your-openai-api-key
LANGSMITH_API_KEY=your-langsmith-api-key
LANGSMITH_PROJECT=your-langsmith-project
```

### 4. Start the API

```sh
uvicorn main:app --reload
```

API will be available at [http://localhost:8000](http://localhost:8000).

---

## Docker Deployment

### Build and run with Docker

```sh
docker build -t support-agent-api .
docker run -d -p 8000:8000 --env-file .env support-agent-api
```

### Or use Docker Compose

```sh
docker compose up --build
```

---

## Running Tests

```sh
pytest tests/
```

Or with Docker Compose:
```sh
docker compose run test

```sh
docker-compose run test
```

---

## How It Works

- **Document Monitoring:** A background thread watches `data/docs/` for changes and updates the vector store automatically.
- **RAG Pipeline:** Queries are answered using both LLM and relevant document context.
- **Fallbacks:** If no relevant context is found, the agent provides a general answer or escalates to a human agent.

---

## Adding/Updating Documents

1. Place new or updated `.txt` or `.pdf` files in `data/docs/`.
2. The background thread will detect changes and update the vector store automatically.

---

## Environment Variables

- `OPENAI_API_KEY` – Required for OpenAI LLM access
- `LANGSMITH_API_KEY` – For LangSmith tracing (optional but recommended)
- `LANGSMITH_PROJECT` – LangSmith project name

---





