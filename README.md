Be sure to have correct Poetry version 1.2.2
Follow: https://python-poetry.org/blog/announcing-poetry-1.2.0/

Follow steps here:
https://doc.courtbouillon.org/weasyprint/stable/first_steps.html

WINDOWS! note: GTK3 installer step ^

install deps

if you have nvidia and cuda on your machine:
```shell
poetry install --with cuda
```
else:
```shell
poetry install
```
OR if you're on linux and have rocm:
```shell
poetry install --with rocm
```

run project with:
```shell
uvicorn main:app --reload  --workers 1 --host 0.0.0.0 --port 8088
```

---

likely not needed anymore, but if above cuda method doesn't work, install with
```shell
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
```