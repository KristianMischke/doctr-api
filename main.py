from typing import Union

import fastapi
from fastapi import FastAPI
import requests

from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from starlette.responses import PlainTextResponse, Response

from receipt_processing import convert_receipt_image_to_df, convert_receipt_image_to_text

model = ocr_predictor(pretrained=True)
app = FastAPI()


class DFResponse(Response):
    media_type = "application/json"

    def render(self, content: any) -> bytes:
        return content.encode("utf-8")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/ocr")
def do_ocr(path: str):
    response = requests.get(path)
    doc = DocumentFile.from_images(response.content)
    # Analyze
    result = model(doc)
    return result


@app.get("/receipt_to_text",
         response_class=PlainTextResponse
         )
def receipt_to_text(path: str):
    response = requests.get(path)
    return convert_receipt_image_to_text(response.content, model, True)


@app.get("/receipt_to_items",
         response_class=DFResponse
         )
def receipt_to_items(path: str):
    response = requests.get(path)
    df = convert_receipt_image_to_df(response.content, model, True)
    return df.to_json()


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
