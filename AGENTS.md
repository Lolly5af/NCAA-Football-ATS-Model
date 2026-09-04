# Agent instructions for CFB ATS Model

## Mission
Maintain a leakage-safe NCAA football ATS model for an ESPN Pick'em pool. Optimize for calibrated probability and stable season-long performance, not sportsbook marketing claims.

## Non-negotiable validation rules
- Never use information that was unavailable before the predicted game.
- Never use final-season ratings, postgame injuries, or future results in a pregame feature.
- Every model change must be evaluated with the existing walk-forward backtest.
- Never fabricate or hand-edit a performance metric.
- Always compare against a simple market/favorite baseline.
- Prefer small, justified changes over feature bloat.

## 2026 operating rules
- Current-season predictions should train only on completed games available before the target week.
- Preserve prediction files so weekly performance can be measured prospectively.
- ESPN pool percentages are decision-support inputs; they must not contaminate the predictive model with future outcomes.
- For Pick'em, choose the side with the higher calibrated cover probability unless a documented pool-risk layer provides a better season-ranking objective.

## When improving the model
1. Inspect the current code and backtest report.
2. Identify a measurable weakness.
3. Make the smallest defensible change.
4. Run tests and the historical walk-forward backtest.
5. Report model vs market baseline and calibration.
6. Do not claim improvement unless the out-of-sample evidence supports it.
