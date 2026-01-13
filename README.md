AI Agent – Omkar Dash

I’m Omkar Dash, with an M.S. in Information Technology from the University of Massachusetts. This project is an early step in my work on agentic AI, focused on building a planning–execution browser agent that can autonomously interact with websites using structured actions such as typing, clicking, and selecting elements.

Despite limited access to paid LLMs, the agent successfully completed multiple Omnizon tasks on the REAL leaderboard, demonstrating the effectiveness of the architecture. With stronger models and more time, I believe the system could achieve broader task coverage and higher reliability.

Repository Note

This project is hosted on my University of Massachusetts GitHub account due to a temporary restriction on my primary GitHub account related to an account-duplication issue. The code here represents the complete and original implementation.

Project Overview

The agent follows a Plan-and-Execute architecture.

Planning Phase

The LLM converts a high-level goal into a JSON-based action plan

Supported actions: fill, click, select

JSON ensures consistent, machine-readable plans and reduces execution errors

Execution Phase

Reads the browser accessibility tree

Matches planned actions to DOM elements and executes them

Uses heuristic fallback and retries when elements are not found

This design keeps the agent reliable even with weaker language models.

Model & Constraints

LLM: deepseek-ai/deepseek-v3.1-terminus

Limited access to paid models

Partial task coverage due to time constraints

Setup
Environment Variables

macOS / Linux

export LLM_API_KEY="your_api_key_here"
export LLM_BASE_URL="https://api.deepseek.com"


Windows (PowerShell)

setx LLM_API_KEY "your_api_key_here"
setx LLM_BASE_URL "https://api.deepseek.com"

Key Points

Agentic AI system with structured planning and execution

JSON-based plans for deterministic behavior

Validated on REAL / Omnizon benchmark tasks

Designed to work under LLM limitations
