# IdentifyChanges
 
> A streaming change / outlier detection **package built for [NovaVision](https://github.com/novavision-ai)**.
 
## Overview
 
**IdentifyChanges** is a NovaVision component that detects distribution shifts and outliers in a stream of embedding vectors. For each incoming embedding it maintains a running baseline (mean, variance, std) of the embeddings and their cosine similarity, then flags samples that deviate significantly from that baseline using a z-score → percentile test.
 
Typical use: monitoring an embedding stream (e.g. per-object or per-frame feature vectors) and raising a flag when the content meaningfully changes.
 
## Pipeline
 
- **Input** — `inputData`: a single embedding or a list of embeddings. List items may be dicts carrying an `embedding` (and optional `uID`) field, or raw embedding vectors.
- **Output** — `outputData`: each item enriched with the detection result — `is_outlier`, `percentile`, `z_score`, the current `average`/`std` baseline, and a `warming_up` flag.
## How It Works
 
1. Normalize each incoming embedding (L2).
2. Compute cosine similarity against the running average embedding.
3. Update the running statistics with the selected strategy (EMA / SMA / SlidingWindow).
4. Convert the similarity into a z-score, then a percentile.
5. Flag as an outlier when the percentile falls into either tail of the configured threshold — but only after the warmup period.
## Strategies
 
The strategy decides *how* the baseline is maintained; the outlier decision is identical for all.
 
| Strategy | Memory | Behavior |
|----------|--------|----------|
| **EMA** — Exponential Moving Average | O(1) | Exponentially decaying weights; recent samples matter more. Tuned by `smoothingFactor`. |
| **SMA** — Simple Moving Average (Welford) | O(1) | Equal weight to all past samples; stable but slow to adapt. |
| **SlidingWindow** | O(N) | Keeps the last N samples in a FIFO buffer; adapts fast. Tuned by `windowSize`. |
 
## Key Configs
 
| Config | Default | Description |
|--------|---------|-------------|
| `Warmup` | 10 | Min samples collected before detection starts (must be ≥ 2). |
| `IdentifyChangesStrategy` | EMA | Baseline strategy: `EMA`, `SMA`, or `SlidingWindow`. |
| `SmoothingFactor` | 0.05 | EMA decay factor α, range `[0.0, 1.0]`. Lower = smoother. |
| `WindowSize` | 10 | Samples kept by SlidingWindow / SMA (must be ≥ 2). |
 
## Structure
 
```
src/
├── executors/IdentifyChanges.py  # Executor: embedding processing, stats update, outlier logic
├── models/PackageModel.py        # Pydantic schemas (input, output, strategies, configs)
└── utils/response.py             # Response builder
```
 
## Install
 
```bash
pip install .
```
 
Requires **Python 3.6+**, NumPy, and the NovaVision `sdk`. Designed to run inside the NovaVision runtime.
 
## License
 
[MIT](LICENSE)
