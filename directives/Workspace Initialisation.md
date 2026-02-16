# Directive: Workspace Initialisation

## Goal

Establish the "Mission Control" environment in Google Antigravity for the **Titan-Oanda-Algo** project.

## Inputs

- **Project Name:** Titan-Oanda-Algo
- **Global Rules:** `.antigravity/rules.md`

## Steps

### 1. Manager Setup

Launch the Antigravity Agent Manager and create the workspace.

### 2. Environment Configuration

- Initialise a `.env` file using `scripts/setup_env.py`.
- Ensure the `.antigravity/` folder is present and contains `rules.md`.

### 3. Agent Allocation

- Verify the presence of the **Architect**, **Engineer**, and **Researcher** agents.
- Synchronise their context windows with the project root.

## Outputs

- Active Antigravity Workspace
- Validated `.env` file with OANDA API credentials