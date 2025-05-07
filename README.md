# PyTorch-based Project Template

## Setup

Docker
```bash
$ cd docker && docker compose up -d --build
$ docker compose down && docker compose up -d --build && docker exec -it [container_name] bash
```

uv
```bash
$ uv python install 3.12

$ uv python pin 3.12
$ uv venv .venv
$ source .venv/bin/activate
$ uv init

## uv add numpy ...
$ uv add ipykernel  # if JupyterLab is used

$ uv lock
## uv sync    # installs everything into .venv
$ uv export --format=requirements.txt > requirements.txt
$ deactivate
```

venv
```bash
$ pyenv install 3.10
$ pyenv local 3.10
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -U pip

## pip install ...

$ pip freeze > requirements.txt
$ deactivate
```

## Usage

## Project Configuration

```bash
.
├── README.md            # プロジェクト概要と使い方を記述
├── conf/                # 実験設定ファイル (例: parameters.yml, secrets.yml)
├── data/                # データや中間成果物の一時保存場所
├── notebooks/           # JupyterLabでの実験ノート
├── pyproject.toml       # Pythonプロジェクトの主要設定ファイル (PEP 518準拠)
├── setup.cfg            # pyproject.toml未対応の設定を補完
└── src/                 # Pythonパッケージコード (共通処理の関数やクラスなど)
```

## Data directory

Please see the details [here](https://docs.kedro.org/en/stable/faq/faq.html#what-is-data-engineering-convention).

```bash
data/
├── 01_raw/              # Original, immutable data from source systems
├── 02_intermediate/     # Partially processed (cleaned/transformed) data
├── 03_primary/          # Canonical datasets for feature engineering
├── 04_feature/          # Engineered features ready for modeling
├── 05_model_input/      # Data prepared specifically for model training
├── 06_models/           # Trained models (e.g., .pkl, .h5 files)
├── 07_model_output/     # Model outputs like predictions or embeddings
└── 08_reporting/        # Reports, visualizations, dashboards, final outputs
```

## Development

### Git Workflow

This project follows `git-flow`.

1.  **Starting Work:** Begin new development or experiments by creating a branch off of the `develop` branch.
2.  **Merging to `develop`:** Once work intended for `develop` is complete, merge it into the `develop` branch (creating a Pull Request on GitHub is recommended if applicable) and delete the original branch.
3.  **Keeping History (Archiving):** If you wish to keep the code and history of a branch *without* merging it into `develop` (e.g., failed experiments, pure explorations), rename the branch to `archive/<original-branch-name>`.
    * Example: To archive a branch named `exp/try-hyperparams-v1`, rename it to `archive/exp/try-hyperparams-v1`.
4.  **Archived Branches:** Branches prefixed with `archive/` are kept for reference only and must *not* be merged into active branches like `develop` or `main`.

### Commit Message

Following commitlint rule: 
- https://github.com/conventional-changelog/commitlint
- https://www.conventionalcommits.org/en/v1.0.0/

### Jupyter Notebook on Cursor Editor

<iframe width="560" height="315" src="https://www.youtube.com/embed/eOSfeBIBzr0?si=MFjxL47thNJGC1SN" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## Reference

- [【Pythonの仮想環境を比較】〜オススメを紹介 〜](https://youtu.be/r4SkIhQThe0?si=kziY5m9s05gCk9Hx)
- [モダンなPyTorchのテンプレ](https://zenn.dev/dena/articles/6f04641801b387)
- [timm – PyTorch Image Models](https://huggingface.co/timm)
- [Best Practices for Python Coding](https://cyberagentailab.github.io/BestPracticesForPythonCoding/)

### Sample Projects

- https://github.com/Delgan/loguru
