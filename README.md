Be sure to have correct Poetry version 1.2.2
Follow: https://python-poetry.org/blog/announcing-poetry-1.2.0/

Follow steps here:
https://doc.courtbouillon.org/weasyprint/stable/first_steps.html

WINDOWS! note: GTK3 installer step ^

install deps
```shell
poetry install
```

install torch if you want to use CUDA locally
```shell
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
```

run project
```shell
uvicorn main:app --reload
```