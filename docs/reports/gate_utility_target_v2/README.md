# LIBERO Gate Utility Target V2 independent validation

This directory is the durable, repository-contained handoff for the preregistered validation of Utility Target V2.

- `report.html`: primary self-contained technical report.
- `artifact.json`: canonical source for the portable report reader.
- `acceptance_results.json`: compact, machine-readable decision evidence and source hashes.
- `VALIDATION.md`: concise engineering interpretation and next-step contract.

The formal result is **GO**, but the scope is deliberately narrow: it authorizes an offline Tiny-MLP feasibility experiment. It does not establish that a learned Gate is predictable, calibrated, compute-efficient, or better in closed-loop LIBERO rollouts.

The report is generated from bounded, reviewed datasets embedded in `artifact.json` and packaged with the Data Analytics portable report builder. `report.html` is self-contained and does not require a server or network access.

Builder validation and packaging passed. The receipt reported `verification=structural_only` because this host has no installed Chromium; semantic chart tables remain embedded, and the five formal PNG figures independently passed visual and scientific QA.
