# Lightsheet movie explorer

A Jupyter notebook for exploring one tracked lightsheet movie from a nuclear segmentation + tracking pipeline: a nuclear channel (e.g. NLS) plus, optionally, one reporter channel (e.g. β-catenin, tbxta-GFP).

It covers metadata and unit checks; per-nucleus QC (brightness, volume, elongation, deep cells) with a QC panel; drift and tracking checks; an aggregate center from nuclear density; distance, speed, persistence, approach velocity, local density and depth; averages over time; reporter distributions and threshold rules with a positive/negative status and onset times; a positive-call image gallery; masks with track tails; and a parquet export with provenance (inputs, settings, git commit) embedded.

It never loads raw frames, so it runs on a 16 GB laptop even for movies with 2+ GB frames.

## Setup (once)

1. Install [uv](https://docs.astral.sh/uv/):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh                  # macOS / Linux
   # powershell -c "irm https://astral.sh/uv/install.ps1 | iex"     # Windows
   ```
2. Get the code and build the environment:
   ```bash
   git clone https://github.com/<briametzger>/lightsheet-movie-explorer.git
   cd lightsheet-movie-explorer
   uv sync        # installs Python 3.12 if needed and the exact package versions in uv.lock
   ```
3. Open the folder in VS Code and select the `.venv` kernel for `movie_explorer.ipynb` (or run `uv run jupyter lab` after `uv add jupyterlab`).

Updating later: `git pull && uv sync`.

## Use

1. Put your pipeline outputs in `data/<dataset>/` (see `data/README.md`). Data is never committed.
2. Open `movie_explorer.ipynb` and edit **only the CONFIG cell**: paths, `EMBRYO_ID`, voxel size and frame interval (confirm from your raw metadata), and `CHANNEL_NAMES` / `NUCLEAR_CH` / `SECOND_CH`.
3. Run top to bottom. Check §1 (frame counts match), §3 (masks and centroids sit on nuclei; the channel check passes), then the QC panel in §4.
4. The reporter threshold has **no default value**: look at the §11 histograms, set `FIXED_THRESHOLD` (or choose `adaptive` / `percentile`), and rerun from §11.

Outputs go to `exports/<dataset>/` (parquet, params JSON, run log, per-embryo summary) and `plots/<dataset>/`.

## Large movies

Mask volumes are read only for image panels, one frame at a time, and cached as 2D label MIPs in `cache/<dataset>/label_mips/`. To precompute all frames (for example on a workstation), run:

```bash
uv run python scripts/make_label_mips.py --masks data/<dataset>/segments.zarr --out cache/<dataset>/label_mips
```

It is resumable and can be split across processes with `--start` / `--stop`. `tools/ram_budget.py` estimates peak memory before heavy steps (`uv run python tools/ram_budget.py zarr data/<dataset>/segments.zarr`).

## Optional

Reading `.czi` MIPs or raw metadata needs `uv add aicsimageio aicspylibczi`.
