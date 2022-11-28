from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Any, Union

import doctr.io
import numpy as np
from doctr.utils import BoundingBox
from pydantic import BaseModel

from starlette.responses import Response


class Geometry(BaseModel):
    x: float
    y: float
    x2: float
    y2: float

    @staticmethod
    def convert(geometry: Union[BoundingBox, np.ndarray]) -> Geometry:
        return Geometry(
            x=geometry[0][0],
            y=geometry[0][1],
            x2=geometry[1][0],
            y2=geometry[1][1],
        )

    def overlaps(self, other: Geometry, overlap_margin: float = 0) -> bool:
        """
            determines whether other overlaps with self
        :param other:
        :param overlap_margin: amount of margin to ignore
        (larger values means geometries must overlap MORE to count as overlapping)
        :return:
        """
        if self.x > other.x2 - overlap_margin:
            return False
        if self.x2 < other.x + overlap_margin:
            return False
        if self.y > other.y2 - overlap_margin:
            return False
        if self.y2 < other.y + overlap_margin:
            return False
        return True


class Word(BaseModel):
    value: str
    confidence: float
    geometry: Geometry

    @staticmethod
    def get_from_doctr_word(word: doctr.io.Word) -> Word:
        return Word(**{
            **word.__dict__,
            'geometry': Geometry.convert(word.geometry)
        })


class Line(BaseModel):
    words: List[Word]
    geometry: Geometry

    @staticmethod
    def get_from_doctr_line(line: doctr.io.Line) -> Line:
        return Line(**{
            **line.__dict__,
            'words': [Word.get_from_doctr_word(word) for word in line.words],
            'geometry': Geometry.convert(line.geometry)
        })


class Artefact(BaseModel):
    type: str
    geometry: Geometry
    confidence: float

    @staticmethod
    def get_from_doctr_artefact(artefact: doctr.io.Artefact) -> Artefact:
        return Artefact(**{
            **artefact.__dict__,
            'geometry': Geometry.convert(artefact.geometry)
        })


class Block(BaseModel):
    lines: List[Line]
    artefacts: List[Artefact]
    geometry: Geometry

    @staticmethod
    def get_from_doctr_block(block: doctr.io.Block) -> Block:
        return Block(**{
            **block.__dict__,
            'lines': [Line.get_from_doctr_line(line) for line in block.lines],
            'artefacts': [Artefact.get_from_doctr_artefact(artefact) for artefact in block.artefacts],
            'geometry': Geometry.convert(block.geometry)
        })


class Page(BaseModel):
    blocks: List[Block]
    page_idx: int
    dimensions: Tuple[int, int]
    orientation: Optional[Dict[str, Any]] = None,
    language: Optional[Dict[str, Any]] = None,

    @staticmethod
    def get_from_doctr_page(page: doctr.io.Page) -> Page:
        return Page(**{
            **page.__dict__,
            'blocks': [Block.get_from_doctr_block(block) for block in page.blocks]
        })


class DFResponse(Response):
    media_type = "application/json"

    def render(self, content: any) -> bytes:
        return content.encode("utf-8")
