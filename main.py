from typing import Union

from fastapi import FastAPI
import requests

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

model = ocr_predictor(pretrained=True)
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/ocr")
def do_ocr(path: str):
    # PDF
    response = requests.get(path)
    doc = DocumentFile.from_images(response.content)
    # Analyze
    result = model(doc)
    return result


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
