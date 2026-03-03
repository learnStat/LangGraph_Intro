# LangGraph Research Pipeline

A multi-step AI research pipeline built with [LangGraph](https://github.com/langchain-ai/langgraph) and GPT-4o-mini. Given any research topic, the pipeline generates focused research questions, answers them with substantive analysis, summarizes the findings, and then critiques and refines the report — all in a single automated workflow.

## How It Works

The pipeline is modelled as a directed graph with four sequential nodes:

```
generate_questions → answer_questions → summarize → critique → END
```

| Node | Role |
|---|---|
| `generate_questions` | Senior research analyst — generates 3 questions covering business impact, technical challenges, and implementation best practices |
| `answer_questions` | Seasoned technology practitioner — provides substantive answers with industry examples, pitfalls, and emerging trends |
| `summarize` | Experienced technology writer — distills answers into a concise, jargon-free summary report |
| `critique` | Hard-nosed industry veteran — identifies gaps, challenges assumptions, and produces a revised final report |

Each node passes its output forward via a shared `ResearchState` TypedDict, making the state explicit and inspectable at every step.

## Output

Two reports are saved to the `outcomes/` directory after each run:

- `outcomes/report.txt` — initial summary
- `outcomes/final_report.txt` — critiqued and revised final report

## Prerequisites

- Python 3.10+
- An OpenAI API key

## Setup

1. Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root with your OpenAI API key:

```
OPENAI_API_KEY=your-key-here
```

3. Create the output directory if it doesn't exist:

```bash
mkdir -p outcomes
```

## Usage

```bash
python src/research_pipeline.py
```

You will be prompted to enter a research topic:

```
Enter a research topic: Generative AI in creative industries
```

The pipeline will then run through all four stages, printing progress as it goes, and save both reports to `outcomes/`.

## Dependencies

| Package | Purpose |
|---|---|
| `langgraph` | Stateful multi-step workflow orchestration |
| `langchain` | LLM abstraction and message formatting |
| `langchain-openai` | OpenAI chat model integration |
| `python-dotenv` | Load API keys from `.env` |
