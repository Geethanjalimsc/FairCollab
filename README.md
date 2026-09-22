
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
