import math
import re

import pandas as pd

from doctr.io import DocumentFile, Page
from doctr.models.predictor import OCRPredictor


def get_points(geometry):
    return geometry[0][0],\
           geometry[0][1],\
           geometry[1][0],\
           geometry[1][1]


def convert_doc_page_to_text_grid(page: Page, debug: bool) -> list[str]:

    # get page geometry as bounding box of all blocks in page
    page_geometry = ((math.inf, math.inf), (-math.inf, -math.inf))
    for block in page.blocks:
        page_geometry = (
            (
                min(page_geometry[0][0], block.geometry[0][0]),
                min(page_geometry[0][1], block.geometry[0][1]),
            ),
            (
                max(page_geometry[1][0], block.geometry[1][0]),
                max(page_geometry[1][1], block.geometry[1][1]),
            )
        )
    if debug:
        print(page_geometry)
    pg_x, pg_y, pg_x2, pg_y2 = get_points(page_geometry)

    all_words = []
    avg_character_widths = []
    for block in page.blocks:
        for line in block.lines:
            for word in line.words:
                x, y, x2, y2 = get_points(word.geometry)
                avg_character_widths.append((x2 - x) / len(word.value))
                all_words.append(word)

    avg_character_widths.sort()
    char_width = avg_character_widths[0]  # char width defined by minimum avg char width
    if debug:
        print(avg_character_widths)
        print(char_width)

    # get number of columns
    cols = math.ceil((pg_x2 - pg_x) / char_width)

    # sort by x then by y
    sorted_words = sorted(all_words, key=lambda word: word.geometry[0][0])
    sorted_words = sorted(sorted_words, key=lambda word: word.geometry[0][1])

    text = []
    ln_start_y = pg_y
    ln_end_y = pg_y
    for word in sorted_words:
        x, y, x2, y2 = get_points(word.geometry)

        new_row = False
        if ln_start_y == ln_end_y:
            # first line
            if debug:
                print('\n', "first line", len(text), ln_start_y, ln_end_y, word, word.geometry, '\n')
            new_row = True

        if y > ln_end_y:
            # below last row
            if debug:
                print('\n', "below last row", len(text), ln_start_y, ln_end_y, word, word.geometry, '\n')
            new_row = True

        if ln_end_y > y > ln_start_y:
            # starts in between the line
            percent_close = (y - ln_start_y) / (ln_end_y - ln_start_y)
            percent_far = (y2 - ln_start_y) / (ln_end_y - ln_start_y)
            if percent_close > 0.5:
                if debug:
                    print('\n', percent_close, "percent_close", len(text), ln_start_y, ln_end_y, word, word.geometry, '\n')
                new_row = True
            elif percent_far > 1.5:
                if debug:
                    print('\n', percent_far, "percent_far", len(text), ln_start_y, ln_end_y, word, word.geometry, '\n')
                new_row = True

        start_col = math.floor((x - pg_x) / char_width)

        if len(text) > 0 and text[-1][start_col:start_col+len(word.value)].strip() != '':
            if debug:
                print('\n', "conflict_new_row", len(text), ln_start_y, ln_end_y, word, word.geometry, '\n')
            new_row = True

        if new_row:
            ln_start_y = y
            ln_end_y = y2

            text.append(" " * cols)

        if text[-1][start_col:start_col+len(word.value)].strip() != '':
            print('\n---')
            print(len(text), ln_start_y, ln_end_y, word, word.geometry)
            print(text[-1])
            print(text[-1][:start_col] + word.value + text[-1][start_col+len(word.value):])
            print(" " * start_col + "^" + "-" * (len(word.value)-1))
            print('---\n')
            assert False

        text[-1] = text[-1][:start_col] + word.value + text[-1][start_col+len(word.value):]

    for row in text:
        print(row)
    return text


def extract_receipt_items(text_rows: list[str]) -> list[dict]:
    items = []

    for line in text_rows:
        stripped = line.strip()
        match = re.search(r"(?P<item>.+?)\s{3,}(?P<price>(\$\s*)?(\d+,?)+\.?\d*)\s*(?P<tag>\w+)?", stripped)
        if match:
            items.append({
                'item': match.group('item'),
                'price': match.group('price'),
                'tag': match.group('tag')
            })

    return items
