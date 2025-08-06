# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TeleQnA is a telecommunications knowledge assessment dataset with 10,000 multiple-choice questions across five categories. The repository contains evaluation scripts to test Large Language Models against this dataset using vLLM local API.

## Setup Commands

Install dependencies:
```bash
pip install -r requirements.txt
```

Extract the password-protected dataset (password: `teleqnadataset`):
```bash
unzip TeleQnA.zip
```

Run evaluation (auto-detects models):
```bash
python run.py
```

Specify model explicitly:
```bash
python run.py your-model-name
```

Use environment variables:
```bash
export VLLM_MODEL="your-model-name"
export VLLM_API_BASE="http://your-server:8000/v1"
python run.py
```

## Key Configuration Requirements

Before running evaluations:
1. Start your vLLM server (default: http://localhost:8000)
2. Ensure `TeleQnA.txt` exists (extract from zip file)

The script will automatically discover available models from your vLLM server using the `/v1/models` endpoint.

## Architecture Overview

**Two-file evaluation system:**
- `run.py`: Main orchestrator that batches questions, handles retries, and tracks results
- `evaluation_tools.py`: vLLM API interface with prompt formatting and response parsing

**Evaluation Flow:**
1. Auto-discover available models from vLLM `/v1/models` endpoint
2. Allow user to select model (or use command line/environment variables)
3. Load questions from `TeleQnA.txt` JSON file
4. Process in configurable batches (default: 5 questions)
5. Send formatted prompts to vLLM API via `check_questions_with_val_output()`
6. Parse and validate responses against ground truth
7. Save incremental results and generate accuracy statistics

**Key Configuration Variables:**
- `n_questions`: Batch size for API calls (default: 5)
- `max_attempts`: Retry limit per batch (default: 5)
- `questions_path`: Dataset file location (default: "TeleQnA.txt")

**Environment Variables:**
- `VLLM_MODEL`: Specify model name directly
- `VLLM_API_BASE`: vLLM server endpoint (default: http://localhost:8000/v1)
- `VLLM_API_KEY`: API key if required (default: "EMPTY")

**Important Technical Notes:**
- Auto-discovers models using `/v1/models` endpoint
- Interactive model selection for multiple available models
- Uses vLLM OpenAI-compatible API with requests library
- Results saved incrementally every 5 batches to `{model}_answers.txt`
- Output includes original dataset plus `tested answer` and `correct` fields
- Accuracy tracking by telecommunications category (Lexicon, Research overview, etc.)