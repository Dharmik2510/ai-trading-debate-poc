# Copilot Coding Agent — Repository Instructions

## Repository Overview
This is the **AI Trading Debate** repository owned by the **Data Science** squad.

## Agent Behaviour for Security Remediation
When assigned a security remediation issue (labelled `automated-remediation`):

1. Read `AGENTS.md` at the repository root — it contains the full workflow and PR format.
2. Parse the CVE table in the issue body to identify affected components and target versions.
3. Locate dependency declarations (`requirements.txt`, `setup.py`, `pyproject.toml`) and apply the minimum version upgrade to fix each CVE.
4. Create a single branch named `copilot/fix-security-<issue_number>`.
5. Open one consolidated PR referencing the issue.

## Stack
- Python (FastAPI backend)
- Jupyter Notebooks

## Key Files
- `requirements.txt` or `pyproject.toml` — Python dependencies
- `src/` — application source code

## Rules
- Never modify `.github/workflows/`, `CODEOWNERS`, or branch protection config.
- Follow conventional commits: `fix(security): upgrade {component} to {version} (CVE-XXXX-XXXXX)`
- Run `pytest` if tests exist; document any failures in the PR.
- See `AGENTS.md` for full PR description format.
