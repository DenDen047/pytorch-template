# Pytorch Project Template

## Usage

```sh
$ cd docker && docker compose up -d --build
$ docker compose down && docker compose up -d --build && docker exec -it [container_name] bash
```

```bash
$ pyenv install 3.12
$ pyenv local 3.12
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -U pip

## pip install ...

$ pip freeze > requirements.txt
$ deactivate
```

## Reference

- [【Pythonの仮想環境を比較】〜オススメを紹介 〜](https://youtu.be/r4SkIhQThe0?si=kziY5m9s05gCk9Hx)
- [モダンなPyTorchのテンプレ](https://zenn.dev/dena/articles/6f04641801b387)
- [timm – PyTorch Image Models](https://huggingface.co/timm)
