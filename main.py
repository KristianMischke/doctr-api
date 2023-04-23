from typing import Union

import pandas as pd
from fastapi import FastAPI, UploadFile
import requests
import torch

from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from starlette.responses import PlainTextResponse

from models import Page, DFResponse
from receipt_processing import convert_doc_page_to_text_grid, extract_receipt_items

model = ocr_predictor(pretrained=True)
app = FastAPI()

if torch.cuda.is_available():
    model = model.cuda()


@app.get("/ocr")
def do_ocr(path: str):
    response = requests.get(path)
    doc = DocumentFile.from_images(response.content)
    # Analyze
    result = model(doc)
    return result


@app.get("/receipt_url_to_text",
         response_class=PlainTextResponse
         )
def receipt_url_to_text(path: str):
    response = requests.get(path, headers={"User-Agent": "XY"})
    print(path, response)
    doc = DocumentFile.from_images(response.content)
    result = model(doc)

    page = Page.get_from_doctr_page(result.pages[0])
    return await receipt_page_to_text(page)


@app.post("/receipt_image_to_text",
         response_class=PlainTextResponse
         )
async def receipt_image_to_text(upload_file: UploadFile):
    doc = DocumentFile.from_images(await upload_file.read())
    result = model(doc)

    page = Page.get_from_doctr_page(result.pages[0])
    return await receipt_page_to_text(page)


@app.post("/receipt_page_to_text",
          response_class=PlainTextResponse
          )
async def receipt_page_to_text(page: Page):
    text_rows = convert_doc_page_to_text_grid(page, True)
    return "\n".join(text_rows)


@app.get("/receipt_url_to_items",
         response_class=DFResponse
         )
async def receipt_url_to_items(path: str):
    response = requests.get(path, headers={"User-Agent": "XY"})
    print(path, response)
    doc = DocumentFile.from_images(response.content)
    result = model(doc)

    page = Page.get_from_doctr_page(result.pages[0])
    return await receipt_page_to_items(page)


@app.post("/receipt_image_to_items",
         response_class=DFResponse
         )
async def receipt_image_to_items(upload_file: UploadFile):
    doc = DocumentFile.from_images(await upload_file.read())
    result = model(doc)

    page = Page.get_from_doctr_page(result.pages[0])
    return await receipt_page_to_items(page)


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


@app.post("/ocr_geometry/upload_image")
async def ocr_geometry(upload_file: UploadFile) -> Page:
    doc = DocumentFile.from_images(await upload_file.read())
    result = model(doc)
    return Page.get_from_doctr_page(result.pages[0])