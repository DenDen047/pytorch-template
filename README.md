# Pytorch Project Template

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

## Reference

- [【Pythonの仮想環境を比較】〜オススメを紹介 〜](https://youtu.be/r4SkIhQThe0?si=kziY5m9s05gCk9Hx)
- [モダンなPyTorchのテンプレ](https://zenn.dev/dena/articles/6f04641801b387)
- [timm – PyTorch Image Models](https://huggingface.co/timm)
- [Best Practices for Python Coding](https://cyberagentailab.github.io/BestPracticesForPythonCoding/)
