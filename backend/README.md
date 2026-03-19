# Analyzer Flask Backend API

A Flask-based REST API for natural language data analysis using Ollama.

## Features

- 🚀 RESTful API with Flask
- 🤖 Integrated with Ollama LLM (gpt-oss:20b)
- 🔒 Secure sandbox execution
- 📊 Automatic DataFrame conversion
- 🌐 CORS enabled for cross-origin requests

## Installation

1. **Install dependencies**:
```bash
cd backend
pip install -r requirements.txt
```

2. **Ensure Ollama is running**:
```bash
ollama serve
ollama pull gpt-oss:20b
```

## Running the Server

```bash
cd backend
python app.py
```

Server will start on `http://localhost:5000`

## API Endpoints

### POST /chat

Submit data and message for analysis.

**Request:**
```json
{
  "data": [
    {"name": "Alice", "age": 25, "salary": 50000},
    {"name": "Bob", "age": 30, "salary": 60000}
  ],
  "jobs": [],
  "message": "What is the average salary?"
}
```

**Response:**
```json
{
  "success": true,
  "response": "61000.0",
  "dataframe_shape": [2, 3],
  "columns": ["name", "age", "salary"]
}
```

### GET /health

Check API and Ollama status.

**Response:**
```json
{
  "status": "healthy",
  "ollama": "running",
  "model": "gpt-oss:20b"
}
```

### GET /

API information and documentation.

## Usage Example

### Using curl:

```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"name": "Alice", "age": 25, "salary": 50000},
      {"name": "Bob", "age": 30, "salary": 60000}
    ],
    "jobs": [],
    "message": "What is the average salary?"
  }'
```

### Using Python:

```python
import requests

response = requests.post(
    "http://localhost:5000/chat",
    json={
        "data": [
            {"name": "Alice", "age": 25, "salary": 50000},
            {"name": "Bob", "age": 30, "salary": 60000}
        ],
        "jobs": [],
        "message": "What is the average salary?"
    }
)

print(response.json())
```

### Using the test script:

```bash
python backend/test_api.py
```

## Request Format

### Required Fields:
- `message` (string): Natural language query
- `data` or `jobs` (array): Data records to analyze

### Optional Fields:
- `jobs` (array): Alternative data source

## Response Format

### Success Response:
```json
{
  "success": true,
  "response": "Analysis result...",
  "dataframe_shape": [rows, cols],
  "columns": ["col1", "col2", ...]
}
```

### Error Response:
```json
{
  "success": false,
  "error": "Error message"
}
```

## Configuration

Edit `app.py` to customize:

- **Port**: Change `port=5000` in `app.run()`
- **Model**: Change `model="gpt-oss:20b"` in OllamaLLM
- **Ollama URL**: Change `base_url="http://localhost:11434"`
- **Sandbox**: Toggle `use_sandbox=True/False`

## Troubleshooting

### Ollama Connection Error
```
Error: Ollama API error: Connection refused
```
**Solution**: Start Ollama server with `ollama serve`

### Model Not Found
```
Error: model 'gpt-oss:20b' not found
```
**Solution**: Pull the model with `ollama pull gpt-oss:20b`

### Import Error
```
Error: No module named 'flask'
```
**Solution**: Install dependencies with `pip install -r requirements.txt`

## Architecture

```
backend/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── test_api.py        # API test script
└── README.md          # This file

Flow:
Client → POST /chat → Flask → Analyzer → Ollama → Response
```

## Security

- ✅ Sandbox execution enabled by default
- ✅ AST validation of generated code
- ✅ Restricted built-in functions
- ✅ Blocked dangerous modules
- ✅ CORS configured for production use

## License

MIT License
