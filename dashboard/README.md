# dashboard/ — Streamlit Visualization Dashboard

Interactive dashboard for comparing training runs, evaluating results, and monitoring experiments.

## Directory Structure

```
dashboard/
├── dashboard.py          # Main Streamlit app
├── pages/                # Multi-page dashboard sections
├── data/                 # Dashboard data cache
├── utils/                # Dashboard-specific utilities
└── requirements.txt      # Additional dependencies
```

## Launch Dashboard

```bash
# Activate conda environment
conda activate mlir

# Launch dashboard
streamlit run dashboard/dashboard.py
```

The dashboard will open at `http://localhost:8501` (or the next available port).

## Features

- **Training progress**: Compare learning curves across runs
- **Eval results**: View speedup distributions and per-benchmark results
- **Baseline comparison**: Compare RL schedules vs MLIR baselines
- **HPO trials**: Analyze hyperparameter optimization results

## Data Sources

The dashboard reads from:
- `results/ops_and_blocks_results/` — Training and eval outputs
- `results/ops_and_blocks_results/baselines/` — Baseline timing data
- `config/ops_and_blocks/` — Config files for run metadata

## Dependencies

```bash
pip install streamlit plotly pandas numpy
```

Or use the provided requirements:
```bash
pip install -r dashboard/requirements.txt
```

## Pages

The dashboard uses Streamlit's multi-page feature. Pages are in `dashboard/pages/`:
- Training comparison
- Evaluation results
- Baseline analysis
- HPO visualization

## Notes

- Dashboard is **read-only** — it doesn't modify any results
- Data is cached in `dashboard/data/` for performance
- Refresh by restarting the Streamlit app
