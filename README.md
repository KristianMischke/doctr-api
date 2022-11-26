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