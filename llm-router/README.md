# Local LLM Router

A router that takes an incoming prompt, **classifies it** by task type
(code / math reasoning / creative / summarization / general chat) and
**complexity** (simple vs. hard), then sends it to the most appropriate
**local model running on Ollama** -- balancing **cost vs. quality**.

It also implements **load balancing** (round-robin across multiple Ollama
instances, if you run more than one) and **failover** (if the chosen
model fails or times out, it automatically retries with a fallback model).

```
                ┌─────────────────────────┐
                │        Your prompt       │
                └────────────┬─────────────┘
                              │
                       ┌──────▼───────┐
                       │ TaskClassifier│  (embedding similarity +
                       │               │   logistic regression)
                       └──────┬───────┘
                              │  task_type, complexity
                       ┌──────▼───────┐
                       │    Router     │  cost-vs-quality policy
                       └──────┬───────┘
                              │  model_key
                       ┌──────▼───────┐
                       │   ModelPool   │  round-robin + fallback chain
                       └──────┬───────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        phi3:mini      llama3.1:8b     qwen2.5-coder:7b   ... (Ollama)
```

---

## 1. Prerequisites

- **Python 3.10+**
- **VS Code** (with the Python extension)
- **Ollama** -- runs the local open-source LLMs
  - Install from https://ollama.com/download (Windows/Mac/Linux installers available)
  - Verify install: `ollama --version`

---

## 2. Project setup

Open a terminal in VS Code (`` Ctrl+` ``) and run:

```bash
# 1. Create & activate a virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 2. Install Python dependencies
pip install -r requirements.txt
```

> The first time `TaskClassifier` runs it will download the
> `all-MiniLM-L6-v2` sentence-transformer model (~80MB) -- you need
> an internet connection for this **one-time** download.

---

## 3. Pull the local LLMs with Ollama

In a terminal, make sure the Ollama service is running, then pull the
models referenced in `config/router_config.yaml`:

```bash
ollama pull phi3:mini
ollama pull llama3.2:3b
ollama pull llama3.1:8b
ollama pull qwen2.5-coder:7b
ollama pull mistral:7b
```

> 💡 **Tip for limited disk/RAM:** you don't need all 5. Start with just
> `phi3:mini` and `llama3.2:3b`, and edit `config/router_config.yaml` so
> every "model role" points at one of those two until you confirm
> everything works end-to-end. Then add the bigger/specialized models.

Ollama runs its API server automatically at `http://localhost:11434`
once installed (run `ollama serve` manually if it's not already running).

---

## 4. Train the complexity classifier

This trains a small logistic-regression model on
`data/training_prompts.csv` that scores prompts from 0 (simple) to 1
(complex):

```bash
python scripts/train_classifier.py
```

This creates `models/complexity_clf.joblib`. (If you skip this step,
the router still works -- it just falls back to a simpler heuristic.)

> 📈 To improve accuracy, add more labeled examples to
> `data/training_prompts.csv` and re-run this script.

---

## 5. Run the router API

```bash
uvicorn app.main:app --reload
```

This starts the FastAPI server at `http://localhost:8000`.
Open the auto-generated docs at `http://localhost:8000/docs` to try
the `/chat` endpoint directly from the browser.

### Quick test with curl

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a Python function to reverse a linked list."}'
```

Example response:

```json
{
  "response": "...",
  "model_used": "qwen2.5-coder:7b",
  "model_key": "code",
  "task_type": "code",
  "complexity": 0.71,
  "latency": 4.32,
  "fallback_used": false
}
```

---

## 6. Run the demo UI (optional)

In a **second** terminal (keep the API running in the first one):

```bash
streamlit run ui/streamlit_app.py
```

This opens a simple chat UI in your browser showing the router's
response plus *which model handled it and why*.

---

## 7. Benchmark / test the router

With the API still running, in another terminal:

```bash
python scripts/benchmark.py
```

This sends a batch of prompts covering every task type and complexity
level, and prints a summary of which models were used, latencies, and
fallback rate.

Check overall stats anytime via:

```bash
curl http://localhost:8000/stats
```

---

## 8. Run unit tests

```bash
pytest
```

These test the **routing decision logic** (`app/router.py`) with mocked
models, so they run instantly without needing Ollama.

---

## Project structure

```
llm-router/
├── README.md
├── requirements.txt
├── config/
│   └── router_config.yaml     # models, Ollama endpoints, fallback chain
├── data/
│   └── training_prompts.csv   # labeled prompts for complexity classifier
├── app/
│   ├── main.py                # FastAPI app (/chat, /health, /stats)
│   ├── classifier.py          # task-type + complexity classification
│   ├── features.py            # feature extraction for complexity model
│   ├── router.py              # cost-vs-quality routing policy
│   ├── model_pool.py          # Ollama calls, load balancing, failover
│   └── logger.py              # JSONL request logging
├── scripts/
│   ├── train_classifier.py    # trains the complexity classifier
│   └── benchmark.py           # sends test prompts, reports stats
├── ui/
│   └── streamlit_app.py        # simple chat UI
└── tests/
    └── test_router.py          # unit tests for routing logic
```

---

## How the routing decision is made

1. **`app/classifier.py`** computes:
   - `task_type` -- via cosine similarity between the prompt's embedding
     and short descriptions of each category (zero-shot, no training data
     needed).
   - `complexity` -- a 0-1 score from a logistic regression model trained
     on word count, sentence count, presence of code, "step by step"
     language, etc.

2. **`app/router.py`** (`select_model_key`) maps `(task_type, complexity)`
   to a **model role** defined in `config/router_config.yaml`:

   | task_type        | complexity        | model role          |
   |-------------------|-------------------|----------------------|
   | code              | any               | `code`               |
   | math_reasoning    | < threshold       | `general`            |
   | math_reasoning    | >= threshold      | `complex_reasoning`  |
   | creative          | any               | `creative`           |
   | summarization     | any               | `general`            |
   | general_chat      | < threshold       | `simple_chat`        |
   | general_chat      | >= threshold      | `general`            |

3. **`app/model_pool.py`** (`generate`) resolves the model role to an
   actual Ollama model tag, picks an Ollama instance (round-robin across
   `base_urls`), and calls `/api/generate`. If that fails, it retries with
   the next model in `fallback_chain`.

---

## Extending this project

- **More categories / finer routing**: add entries to `TASK_CATEGORIES`
  in `app/classifier.py` and corresponding model roles in
  `config/router_config.yaml`.
- **True load balancing**: run a second Ollama instance
  (`OLLAMA_HOST=127.0.0.1:11435 ollama serve`) and add its URL to
  `ollama.base_urls` in the config.
- **Cost tracking**: extend `app/logger.py` to estimate cost based on
  tokens generated × a per-model "cost weight" you define.
- **Better complexity classifier**: replace the small CSV with a larger
  labeled dataset, or swap logistic regression for a gradient-boosted
  tree (e.g. `sklearn.ensemble.GradientBoostingClassifier`).
