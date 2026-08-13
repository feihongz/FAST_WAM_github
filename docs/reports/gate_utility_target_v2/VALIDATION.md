# Utility Target V2 validation result

## Decision

**GO — proceed to offline Tiny-MLP feasibility only.** All 21 preregistered GO checks passed on the same 100 states using independent validation seeds 47–50 against the frozen five-seed target from seeds 42–46.

The strongest evidence is:

- target-mean versus validation-mean Spearman `rho = 0.707` with stratified-bootstrap 95% interval `[0.555, 0.844]`;
- Kendall `tau = 0.587`, bootstrap interval `[0.459, 0.713]`;
- actionable sign retention `67/83 = 80.7%`;
- high-confidence sign retention `21/24 = 87.5%`, including `12/12` positive and `9/12` negative;
- top/bottom-20% recall `70% / 75%` and Jaccard `53.8% / 60.0%`;
- aggregate nine-seed reliability `ICC(1,9) = 0.828`.

## What was validated

For each LIBERO demonstration state, `E0` and `E10` are valid-step, per-action-dimension MSE values in normalized action space. Utility is `U = E0 - E10`. The target is each state's mean over seeds 42–46; independent validation is the mean over seeds 47–50. Both routes use the same state and paired inference seed. The 400-record validation grid is complete, contains no errors, and passed an independent provenance, route, pairing, and hash audit.

High confidence means the target mean lies outside `+/-1e-4`, at least four of five target seeds agree with that direction, and the two-sided 95% t interval lies wholly beyond the same deadband boundary.

## Interpretation boundary

This validates the **label target sufficiently to test offline learnability**. It does not show that state features can predict utility, that a threshold is calibrated, that the router saves wall-clock compute, or that it improves closed-loop success.

The all-nine aggregate reliability is good, but single-seed reliability remains only `ICC(1,1) = 0.349`. Training must therefore use the multi-seed aggregate and carry uncertainty; it must not fall back to one stochastic utility measurement per state.

## Next engineering step

1. Run a small offline Tiny-MLP feasibility experiment on the current 100-state panel, using frozen visual/text/proprio features and group-aware train/validation splits.
2. Predict continuous mean utility with Huber loss; use uncertainty as sample weight or filtering, and report Spearman, sign retention, high-confidence ranking, and top/bottom precision.
3. Add task/suite-only shortcut baselines. Do not accept a Gate that merely recovers task identity.
4. In parallel, collect seeds 42–46 for the remaining Pilot-500 states so the definitive training set has the same Target V2 definition.
5. Only after held-out offline success, run a small closed-loop route smoke test against fixed-0, fixed-10, and matched-compute random routing.

The exact source hashes, preregistered checks, audit status, and bounded result rows are preserved in `acceptance_results.json` and the interactive `report.html`.
