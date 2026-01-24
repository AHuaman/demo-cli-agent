# CLI Appointment Agent

A high-quality demo of a personal AI assistant that operates in your terminal, helping you manage appointments with intelligent reasoning and built-in observability.

## 🚀 Quickstart

### 1. Prerequisites
- [uv](https://github.com/astral-sh/uv) installed.
- An OpenAI API Key.

### 2. Setup
Clone the repo and create your `.env` file:
```bash
cp .env.example .env
```
Edit `.env` and add your `OPENAI_API_KEY`.

### 3. Install Dependencies
```bash
uv sync
```

### 4. Run the Agent
```bash
uv run appointment-agent
```

### 5. View Traces (MLflow)
In a separate terminal, launch the MLflow UI to see the agent's "thoughts" and tool calls:
```bash
uv run ml-ui
```
Visit `http://localhost:5050` in your browser.

---

## ✨ Features

- **Intelligent Intent Recognition**: Register, search, or delete appointments using natural language.
- **Safety Confirmation**: The agent always asks for your permission before creating or deleting data.
- **Smart Conflict Resolution**: If you try to double-book, a specialized subagent analyzes your schedule and suggests optimal alternative slots.
- **Observability**: Every interaction is traced automatically via MLflow.
- **Modern Stack**: Built with PydanticAI, SQLModel, Typer, and Rich.

## 🏗️ Architecture & Patterns

This project follows several modern Python best practices:
- **Multi-Agent Delegation**: A Supervisor agent delegates complex reasoning (like conflict resolution) to specialized subagents.
- **Dependency Injection**: Tools use `RunContext` to access resources like database sessions.
- **Type Safety**: Pydantic models validate all data flowing between the LLM and the database.
- **Autologging**: Seamless integration with MLflow for debugging and evaluation.

For more details, see [ARCHITECTURE.md](./docs/ARCHITECTURE.md).

## 🧪 Testing
Run the full test suite with:
```bash
uv run pytest
```
The suite covers settings, database integrity, and agent tool logic.
