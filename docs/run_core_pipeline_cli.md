# Core Pipeline CLI Guide

This guide summarizes the command-line options for running the core pipeline (`python run_core_pipeline.py`). Use it as a quick reference when selecting specific steps, adjusting data coverage, or running diagnostic modes.

## Quick Start
- Run the full end-to-end pipeline (steps 1–11):  
  `python run_core_pipeline.py`
- Execute only data ingestion and preprocessing:  
  `python run_core_pipeline.py --steps 1-2`
- Skip data ingestion because the parquet files are already prepared:  
  `python run_core_pipeline.py --skip 1`
- Preview which steps would run without executing them:  
  `python run_core_pipeline.py --dry-run`

## Step Numbers
| Step | Description |
| --- | --- |
| 1 | Data Ingestion (OHLCV + Fundamentals) |
| 2 | Data Preprocessing & Alignment |
| 3 | Technical Feature Engineering |
| 4 | Fundamental Feature Engineering (PIT) |
| 5 | Label Generation |
| 6 | Feature Selection & Normalization |
| 7 | Model Training |
| 8 | Signal Generation (Scoring) |
| 9 | Portfolio Construction |
| 10 | Backtesting |
| 11 | Model Evaluation & Reporting |

Use comma-separated values to run specific steps (e.g., `--steps 1,3,5`) or ranges with a dash (e.g., `--steps 3-7`). Combine `--steps` with `--skip` when you need more granular control; the skip list is applied after step ranges are expanded.

## Core Options
| Flag | Description |
| --- | --- |
| `--steps` | Steps to run. Accepts comma-separated values or ranges. Default: `1-11`. |
| `--skip` | Steps to exclude after expansion. Example: `--skip 1,2` skips data ingestion and preprocessing. |
| `--symbols` | Number of symbols pulled from the S&P 500 universe (top by market cap). Default: `500`. |
| `--start-date` | Override the ingestion start date (YYYY-MM-DD). Falls back to `config/config.yaml` if omitted. |
| `--config` | Alternate configuration file path. Default: `config/config.yaml`. |
| `--continue-on-error` | Continue to the next step even if the current step fails. |
| `--dry-run` | Print the execution plan without running it. Helps confirm step selections. |
| `--validate-only` | Skip execution and only validate required artifacts produced by the core pipeline. |
| `--verbose` | Enable DEBUG-level logging for deeper inspection. |

## Tips
- Use `--dry-run` before long runs to confirm the steps the pipeline will execute.
- When switching price providers (e.g., yfinance vs. SimFin), adjust `ingest.provider` in `config/config.yaml`. Step 1 will honor the new provider automatically.
- If you frequently run the same subset of steps, consider scripting the command (e.g., shell alias or Makefile target) for faster reuse.
