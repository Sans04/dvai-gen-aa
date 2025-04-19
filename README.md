# Diagnosis Veritatis AI - Agent Architecture Backend

This repository implements a state-based LangGraph agent architecture to realise a RAG workflow in which specialised agents work together in a coordinated manner to enable informed and adaptive responses from an LLM through external knowledge retrieval.

## Installation

First, install [Ollama](https://ollama.com), which is used to efficiently run local LLMs and seamlessly integrate them into the agent workflow.

```bash
curl -fsSL https://ollama.com/install.sh | sh
```
Then, install and start the desired LLM in Ollama (e.g. `gemma3`):
```bash
ollama run gemma3:4b
```
> Type /bye in the Ollama terminal to gracefully stop the model.

Clone this repository and set up a virtual Python environment:
```bash
git clone https://github.com/Sans04/dvai-gen-aa.git
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage
Start the virtual Python environment (if not already active):
```bash
source venv/bin/activate
```
Then start the API server for the agent architecture:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
You can now send a question to the LLM using:
```bash
curl -N -X POST https://localhost:8000/invoke -H "Content-Type: application/json" -d '{"question": "Tell me something inspiring."}'
```
> Note: Make sure an Ollama-compatible model (e.g. `gemma3`) is installed.
## License

[MIT](https://choosealicense.com/licenses/mit/)