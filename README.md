# Self Surprise

Experimental PyTorch character language model for studying log-decayed causal
attention, uncertainty prediction, and fast correction adapters.

The main experiment lives in `train.py`. It trains a small GPT-style character
model and logs diagnostics around:

- standard causal attention vs learned log-decayed attention
- generated self-surprise and entropy
- uncertainty heads for surprise, entropy, and change prediction
- EMA teacher diagnostics and optional self-distillation
- fast adapter behavior under high-surprise or high-stress tokens
- nested fast adaptation on generated pseudo-traces

## Setup

Use Python with PyTorch installed. Optional plotting uses `matplotlib`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch matplotlib
```

## Data

`train.py` reads a plain text corpus from `config["input_path"]`. Update that
path near the top of `train.py` before running, or keep a local text corpus at
the configured location.

The script splits the corpus into train and validation segments using
`config["val_frac"]`.

## Run

```bash
python train.py
```

For a quick smoke run:

```bash
SELF_SURPRISE_MAX_STEPS=2 \
SELF_SURPRISE_EVAL_INTERVAL=1 \
SELF_SURPRISE_EVAL_ITERS=1 \
SELF_SURPRISE_BATCH_SIZE=1 \
SELF_SURPRISE_ENABLE_NESTED_ADAPTATION=false \
python train.py
```

## Outputs

By default, training writes artifacts under
`checkpoints_log_decay_self_surprise/`:

- `metrics.csv` for training, validation, adapter, and generation diagnostics
- best model checkpoints as `.pt` files
- post-training plots in `plots/` when plotting is enabled
- optional attention evolution video when enabled
- `final_summary.json` after training completes

## Configuration

Most settings are in the `config` dictionary at the top of `train.py`.

Supported environment overrides include:

- `SELF_SURPRISE_MAX_STEPS`
- `SELF_SURPRISE_EVAL_INTERVAL`
- `SELF_SURPRISE_EVAL_ITERS`
- `SELF_SURPRISE_BATCH_SIZE`
- `SELF_SURPRISE_BLOCK_SIZE`
- `SELF_SURPRISE_ENABLE_NESTED_ADAPTATION`
- `SELF_SURPRISE_NESTED_ADAPT_INTERVAL`
- `SELF_SURPRISE_NESTED_ADAPT_PROMPTS`
- `SELF_SURPRISE_NESTED_INNER_STEPS`
- `SELF_SURPRISE_NESTED_PROMPT_LEN`
- `SELF_SURPRISE_NESTED_ROLLOUT_TOKENS`
- `SELF_SURPRISE_NESTED_UNKNOWN_SOURCE`
- `SELF_SURPRISE_NESTED_SHIFT_MODE`
- `SELF_SURPRISE_NESTED_SHIFT_STRENGTH`
- `SELF_SURPRISE_ENABLE_NESTED_ABLATION_RUNNER`
- `SELF_SURPRISE_NESTED_ABLATION_ROOT`

## License

MIT. See `LICENSE`.
