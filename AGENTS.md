# Data_mining_2024

Datasets, scripts, and slides for the Data Mining 2024 course. Per-experiment
folders under `datasets/`, each holding a task description, a Python script,
and its data.

## Repository Structure

- `datasets/exp1..exp5/` — weekly experiments. Each contains:
  - `expN.md` — task description (Chinese)
  - `parse.py` — solution script (pandas / sklearn / matplotlib)
  - data files: `train.csv` / `test.csv`, or image datasets (`.jpg` / `.xml` / `.txt`)
- `datasets/final/` — final project, split into six sub-projects (区域分割 / 台风预测 / 模型对比 / 特征选择 / 福字识别 / 飞机检测)
- `demos/` — LaTeX report template (`report.tex`) and PDFs
- `slides/` — course slide PDFs (Chinese)

## Conventions

- Task descriptions and slide content are in Chinese; do not translate them.
- Solutions are single-file Python scripts named `parse.py`, runnable from
  their own directory (`os.chdir` is used inside the script).
- Use pandas for data handling, sklearn for modeling, matplotlib for plots.
- Reports are written in LaTeX using the `demos/report.tex` template.
- The repo intentionally contains large datasets (~160 MB). Be mindful of repo
  size: do not commit new generated outputs (`output/`, figures, intermediate
  files) unless required.

## Commands

- Run an experiment script: `python parse.py` (inside its own directory)
- No package manager or test suite; verify changes by running the affected
  `parse.py` and confirming it produces expected output.

## Git Workflow

- Follow the global conventions: PR-driven flow, Conventional Commits, commit
  with the `opencode[bot]` identity, never push directly to `main`.
- One experiment / one coherent change per commit.
