# FairCollab

![Python](https://img.shields.io/badge/Python-3.11.9-blue?logo=python)
![License](https://img.shields.io/badge/Licence-MIT-green)
![Status](https://img.shields.io/badge/Status-Active%20Development-orange)
![MSc Dissertation](https://img.shields.io/badge/MSc%20Dissertation-University%20of%20Birmingham-purple)

An AI-powered multi-agent system for fair, evidence-grounded assessment of
individual contribution in group projects.
MSc dissertation project, University of Birmingham.

---

## Key Features

- **Evidence-grounded assessment** — reasons over real GitHub commits, task
  records, and peer feedback rather than subjective peer opinion alone
- **Five-node LangGraph pipeline** — Retrieve → Integrity Analysis →
  Factor Analysis → Overall Assessment → Validate, with a self-correction
  loop that rewrites unverified claims before returning results
- **RAG retrieval layer** — FAISS + Gemini embeddings with separate retrieval
  paths for contributions and peer feedback, preventing high-volume
  contributors from crowding out peer feedback evidence
- **Six assessment factors** — Quantity, Quality, Type Coverage, Consistency,
  Reliability, Collaboration & Support
- **Integrity layer** — three deterministic checks for corroboration
  confidence, cross-student duplicate detection, and mutual-isolation flagging
- **Fairness-by-design** — non-code contributions (writing, meetings,
  documentation) are explicitly weighted equally to code; verified empirically
  on two structurally different project types

---

## The Problem

Existing peer assessment tools (Buddycheck, CATME, Kritik, FeedbackFruits)
rely on subjective peer opinion, with no independent verification of what a
student actually did. A confident, well-liked student can be rated highly
regardless of real contribution, and a quiet but substantial contributor can
be rated poorly.

FairCollab instead retrieves and reasons over real evidence — GitHub commits,
task records, peer feedback — and produces a cited, written justification for
every rating, so an assessment can be checked against the record it claims to
be based on.

---

## Architecture Overview

**RAG retrieval layer** (`src/rag_pipeline.py`) — FAISS + Gemini embeddings
(`models/gemini-embedding-001`). `retrieve_evidence()` treats contributions
and peer feedback separately rather than as one combined search, so a student
with a high volume of logged contributions can't crowd low-volume peer
feedback out of the retrieved evidence set: contributions are a ranked, capped
`similarity_search()` (`k=8` by default), while peer feedback is retrieved in
full via a direct, unranked metadata filter over the index.

**LangGraph five-node pipeline** (`src/agent.py`):

Retrieve Evidence → Integrity Analysis → Factor Analysis → Overall Assessment → Validate


`Validate` fact-checks every claim in the assessment against the retrieved
evidence and student stats. If a claim isn't fully supported, it routes
control back to `Factor Analysis` for a revision (capped at
`MAX_REVISIONS = 2`) rather than returning an unverified assessment.

**Six assessment factors** (`src/prompts.py`): Quantity of Contribution,
Quality of Contribution, Contribution Type Coverage, Consistency, Reliability,
Collaboration & Support.

**The fairness principle** — an explicit instruction embedded directly in
`FACTOR_PROMPT` (`src/prompts.py`) stating that non-code contributions
(writing, meetings, documentation, presentations) are equally valuable to
code, and that a factor must never be rated lower merely for not being code.
Verified empirically across two structurally different project types: a
student with zero code contributions (Student D, software project) and a
student with zero writing contributions (Student H, research project) were
both independently rated High overall, with the model's own reasoning
explicitly crediting their non-code work.

**Integrity layer** — three empirically-calibrated checks, all deterministic
(no LLM sampling involved), run inside `node_integrity_analysis()`:

- **Corroboration confidence** (`assess_corroboration_confidence()`) — a
  peer_support claim of helping another student is checked against that
  student's own record, producing one of three tiers (`strong` / `weak` /
  `none`) rather than a binary pass/fail, since collapsing an
  ambiguous-but-plausible case to a hard true/false was found to hide
  genuinely uncertain claims.
- **Cross-student contribution-duplication detection**
  (`find_cross_student_similar_contributions()`) — flags near-duplicate
  contribution descriptions across different students, with a type-aware
  similarity threshold: meeting-type contributions are excluded from flagging
  entirely, since real students independently describing the same meeting
  scored higher in testing than genuine non-meeting duplicates did.
- **Mutual-isolation / zero-corroborators detection**
  (`build_mention_graph()`, `find_mutual_isolation_pairs()`) — a directed
  graph of who mentions/credits whom, used to flag pairs who exclusively
  corroborate each other with no outside confirmation, and students with no
  corroborating mentions at all from anyone in the project.

**Streamlit dashboard** (`src/app.py`) — radar chart of the six factors, an
at-a-glance factor-rating summary row, expandable evidence cards (grouped per
factor, tagged by record type), and a severity-aware integrity section that
distinguishes low-confidence signals from stronger ones rather than presenting
every flag identically.

---

## Project Structure

src/
├── agent.py LangGraph pipeline: five nodes, integrity checks, self-correction loop
├── app.py Streamlit UI: assessment dashboard + Live Data sidebar
├── prompts.py LLM prompt templates (factor analysis, overall rating, validation)
├── rag_pipeline.py FAISS index build/search, evidence retrieval, corroboration/duplicate checks
└── connectors/
├── init.py
├── github_connector.py Fetches commits via the GitHub REST API into contribution records
└── forms_connector.py Reads Google Forms responses via Sheets API

data/
├── students.json Simulated software project (zero-code fairness test case)
├── research_students.json Simulated research project (zero-writing fairness test case)
├── robotics_project.json Live data project, populated by GitHub and Google Forms connectors
└── fraud_scenarios.json Four constructed scenarios with known ground truth


> Two runtime files live under `data/` but are gitignored:
> `faircollab_index/` (built FAISS index) and `live_config.json`
> (saved Live Data connector settings).

---

## Setup and Installation

**Requirements:** Python 3.11.9

```bash
# Clone the repository
git clone https://github.com/Geethanjalimsc/FairCollab.git
cd FairCollab

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Credentials

| What | Where it goes | Required for |
|---|---|---|
| Gemini API key | `.env` as `GOOGLE_API_KEY=...` | Everything — embeddings and LLM calls |
| GitHub Personal Access Token | Entered in the Streamlit sidebar (Live Data → GitHub tab); never written to disk | GitHub connector only |
| Google service account JSON | File on disk (default: `google_service_account.json`, configurable in sidebar) | Google Forms/Sheets connector only |

Create a `.env` file in the project root:

GOOGLE_API_KEY=your_gemini_api_key_here


---

## Running the App

```bash
streamlit run src/app.py
```

The sidebar's **Project Type** selector switches between:
- **Software Project** / **Research Project** — pre-built simulated datasets,
  selectable immediately
- **Live Data** — the real robotics project, populated on demand from GitHub
  and Google Forms via the connectors (the Live Data section only appears
  once this option is selected)

---

## Data

| File | Description |
|---|---|
| `students.json` | Simulated software project. Includes zero-code fairness test case. |
| `research_students.json` | Simulated research project. Includes zero-writing fairness test case. |
| `robotics_project.json` | Live data, populated by GitHub and Google Forms connectors. |
| `fraud_scenarios.json` | Four scenarios with documented ground truth: clean, peer inflation, temporal anomaly, coordinated fraud. |

---

## Academic Context

**Geethanjali Muddahanumaiah**
MSc Artificial Intelligence and Machine Learning
University of Birmingham
Supervisor: Dr Jian Liu

---

## Licence

This project is licensed under the MIT Licence.
See [LICENSE](LICENSE) for details.

> This repository is read-only.
> Feel free to clone it for personal use or open an issue to leave feedback.
> Direct contributions are not accepted at this time.
