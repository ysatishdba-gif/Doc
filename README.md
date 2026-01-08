# Clinical Intent Extraction API

Extract structured clinical intents from medical text. Supports comma-separated inputs for batch processing.

## Project Structure

```
clinical-intent-api/
├── app/
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── logging_config.py  # Logging setup
│   └── pipeline.py        # Core extraction logic
├── prompts/
│   ├── __init__.py
│   ├── query_expansion_prompt.py    # v1.0.0
│   └── intent_extraction_prompt.py  # v1.0.0
├── tests/
│   ├── __init__.py
│   └── test_api.py        # API tests
├── logs/                  # Log files
├── main.py               # FastAPI application
├── requirements.txt
├── Dockerfile
└── README.md
```

## Quick Start

### 1. Set Environment Variables

```bash
export GCP_PROJECT_ID="your-project-id"
export GCP_LOCATION="us-central1"
export MODEL_VERSION="gemini-2.0-flash-001"
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the API

```bash
python main.py
```

Or with uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/version` | GET | API and prompt versions |
| `/extract` | POST | Extract intents |
| `/docs` | GET | Swagger UI |

## Input Format

**Single text:**
```json
{
  "text": "Pt with HTN and DM"
}
```

**Multiple texts (comma-separated):**
```json
{
  "text": "Pt with HTN, SOB on exertion, Family hx of CAD"
}
```

**With metadata:**
```json
{
  "text": "Pt with HTN, SOB on exertion",
  "metadata": {
    "source": "clinical_note",
    "source_id": "NOTE-001"
  }
}
```

## Output Format

```json
{
  "request_id": "uuid",
  "total_inputs": 2,
  "results": [
    {
      "input_index": 0,
      "original_query": "Pt with HTN",
      "expanded_query": "Patient with Hypertension",
      "abbreviations_expanded": ["HTN → Hypertension"],
      "concepts_added": [],
      "is_clinical": true,
      "rejected_reason": null,
      "total_intents": 1,
      "intents": [
        {
          "intent_title": "Hypertension Management",
          "description": "...",
          "nature": "Current Clinical Presentation / Chronic Condition",
          "sub_nature": [...],
          "final_queries": ["hypertension", "high blood pressure", ...]
        }
      ],
      "processing_time_seconds": 1.5
    },
    {
      "input_index": 1,
      "original_query": "SOB on exertion",
      "expanded_query": "Shortness of breath on exertion",
      ...
    }
  ],
  "total_intents_all": 3,
  "total_processing_time_seconds": 3.2,
  "timestamp": "2024-01-15T10:30:00Z",
  "prompt_versions": {
    "query_expansion": "1.0.0",
    "intent_extraction": "1.0.0"
  }
}
```

## Example Usage

### cURL

```bash
# Single input
curl -X POST "http://localhost:8000/extract" \
  -H "Content-Type: application/json" \
  -d '{"text": "Pt with HTN and DM"}'

# Multiple inputs (comma-separated)
curl -X POST "http://localhost:8000/extract" \
  -H "Content-Type: application/json" \
  -d '{"text": "Pt with HTN, SOB on exertion, Family hx of heart disease"}'
```

### Python

```python
import requests

# Single input
response = requests.post(
    "http://localhost:8000/extract",
    json={"text": "Pt with HTN and DM"}
)
print(response.json())

# Multiple inputs
response = requests.post(
    "http://localhost:8000/extract",
    json={"text": "Pt with HTN, SOB on exertion, Diabetes follow-up"}
)
for result in response.json()["results"]:
    print(f"Input {result['input_index']}: {result['total_intents']} intents")
```

## Logging

Logs are written to `logs/app.log` and console. Log entries include:

- **REQUEST**: Input received with text count and source
- **INTENTS**: Intents detected per text
- **AMBIGUOUS**: Non-clinical or ambiguous cases flagged
- **ERROR**: Processing errors
- **RESPONSE**: Response summary with total intents and time

Example log:
```
2024-01-15 10:30:00 | INFO     | REQUEST | id=abc123 | texts_count=3 | source=clinical_note
2024-01-15 10:30:01 | INFO     | INTENTS | id=abc123 | text_idx=0 | intents_count=2 | is_clinical=True
2024-01-15 10:30:02 | INFO     | INTENTS | id=abc123 | text_idx=1 | intents_count=1 | is_clinical=True
2024-01-15 10:30:02 | WARNING  | AMBIGUOUS | id=abc123 | text_idx=2 | reason=Non-clinical
2024-01-15 10:30:02 | INFO     | RESPONSE | id=abc123 | total_intents=3 | processing_time=2.15s
```

## Docker

### Build

```bash
docker build -t clinical-intent-api .
```

### Run

```bash
docker run -p 8000:8000 \
  -e GCP_PROJECT_ID=your-project \
  -e GCP_LOCATION=us-central1 \
  -e MODEL_VERSION=gemini-2.0-flash-001 \
  clinical-intent-api
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=app --cov-report=term-missing
```

## Prompt Versions

| Prompt | Version | File |
|--------|---------|------|
| Query Expansion | 1.0.0 | `prompts/query_expansion_prompt.py` |
| Intent Extraction | 1.0.0 | `prompts/intent_extraction_prompt.py` |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GCP_PROJECT_ID` | - | GCP project ID (required) |
| `GCP_LOCATION` | us-central1 | GCP region |
| `MODEL_VERSION` | gemini-2.0-flash-001 | Model version |
| `API_HOST` | 0.0.0.0 | API host |
| `API_PORT` | 8000 | API port |
| `LOG_LEVEL` | INFO | Logging level |
| `DEBUG` | false | Debug mode |
