from typing import Union

import pandas as pd
from fastapi import FastAPI
import requests

from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from starlette.responses import PlainTextResponse

from models import Page, DFResponse
from receipt_processing import convert_doc_page_to_text_grid, extract_receipt_items

model = ocr_predictor(pretrained=True)
app = FastAPI()


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


@app.get("/receipt_image_to_text",
         response_class=PlainTextResponse
         )
def receipt_image_to_text(path: str):
    response = requests.get(path, headers={"User-Agent": "XY"})
    print(path, response)
    doc = DocumentFile.from_images(response.content)
    result = model(doc)

    text_rows = convert_doc_page_to_text_grid(result.pages[0], True)
    return "\n".join(text_rows)


@app.post("/receipt_page_to_text",
          response_class=PlainTextResponse
          )
async def receipt_page_to_text(page: Page):
    text_rows = convert_doc_page_to_text_grid(page, True)
    return "\n".join(text_rows)


@app.get("/receipt_image_to_items",
         response_class=DFResponse
         )
async def receipt_image_to_items(path: str):
    response = requests.get(path, headers={"User-Agent": "XY"})
    print(path, response)
    doc = DocumentFile.from_images(response.content)
    result = model(doc)

    text_rows = convert_doc_page_to_text_grid(result.pages[0], True)
    items = extract_receipt_items(text_rows)
    df = pd.DataFrame(items)
    return df.to_json()


@app.post("/receipt_page_to_items",
          response_class=DFResponse
          )
async def receipt_page_to_items(page: Page):
    text_rows = convert_doc_page_to_text_grid(page, True)
    items = extract_receipt_items(text_rows)
    df = pd.DataFrame(items)
    return df.to_json()


@app.get("/ocr_geometry")
async def ocr_geometry(path: str) -> Page:
    response = requests.get(path, headers={"User-Agent": "XY"})
    print(path, response)
    doc = DocumentFile.from_images(response.content)
    result = model(doc)
    return Page.get_from_doctr_page(result.pages[0])


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
