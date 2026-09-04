# CFB ATS Pick'em Model

Automated, leakage-safe NCAA FBS against-the-spread model for an ESPN Pick'em pool. The repository is designed to refresh throughout the 2026 season, run walk-forward backtests, and publish weekly predictions without requiring local Python execution.

## Architecture

- CollegeFootballData (CFBD) supplies historical/current games, results, and betting lines.
- Feature construction is pregame-only: dynamic Elo, recent margin, offensive/defensive proxies, home/neutral status, rest, and early-season shrinkage.
- An ensemble combines regularized logistic ATS classification, gradient boosting, and a predicted-margin model.
- Walk-forward validation prevents future information from leaking into predictions.
- GitHub Actions handles scheduled refreshes; no local Python run is required after setup.
- Optional ESPN pool percentages can be supplied in `data/espn_pool.csv` for leverage analysis.

## Important validation rule

The repository must never fabricate a backtest result. `reports/BACKTEST_STATUS.md` remains explicit until a real historical run has completed. Public CFB modeling work has shown why in-sample ATS claims can be badly overstated; this project therefore reports only true out-of-sample results.

## Local development

```bash
python -m pip install -r requirements.txt
python -m src.pipeline --build-history
python -m src.backtest --start-season 2023 --end-season 2025
python -m src.pipeline --current-season 2026
python -m src.predict --season 2026 --week 1
```

## GitHub Actions setup

Add a repository Actions secret named `CFBD_API_KEY`. GitHub Actions secrets are encrypted and are only exposed to workflows that explicitly reference them. See the GitHub Actions secrets documentation.

The scheduled workflow refreshes the current season three times per week and can also be launched manually. The deterministic model execution happens in GitHub Actions, so you do not need to run Python yourself.

## Codex workflow

The repository also includes a GitHub Agentic Workflow definition intended for Codex-assisted maintenance. GitHub Agentic Workflows supports selecting Codex as the engine and authenticating it with `CODEX_API_KEY` or `OPENAI_API_KEY`. The agent is instructed to inspect backtest evidence, improve the model only when justified, run tests, and avoid fabricating performance.

## Model objective

This is optimized for ESPN Pick'em rather than sportsbook ROI. The primary decision is the side with the higher calibrated probability of covering. The model also records market edge and confidence so weekly selections can be ranked consistently.

For the user's season-long top-half-average objective, the eventual next layer is pool simulation using ESPN public-pick percentages. That should influence risk management without overriding the underlying calibrated probabilities.

## Data and reports

- `data/espn_pool.csv` — optional ESPN public-pick percentages.
- `data/processed/` — generated training data; ignored from source control when large.
- `reports/backtest_summary.csv` — generated OOS metrics.
- `reports/backtest_predictions.csv` — generated game-level OOS predictions.
- `reports/YYYY_week_N_picks.csv` — generated weekly card.
- `reports/BACKTEST_STATUS.md` — honest validation status.

## Current status

The code is installed in this repository, but a genuine historical backtest still requires a CFBD API key in GitHub Actions. Once the secret is present, run the workflow manually to generate the first real validation report.
