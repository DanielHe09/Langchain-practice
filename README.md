# FastAPI Server

A simple FastAPI server setup.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Server

Start the server with uvicorn:

```bash
uvicorn main:app --reload
```

The server will be available at:
- API: http://localhost:8000
- Interactive API docs: http://localhost:8000/docs
- Alternative API docs: http://localhost:8000/redoc

## Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /api/items/{item_id}` - Example endpoint with parameters
