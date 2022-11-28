from __future__ import annotations

from thefuzz import fuzz
import numpy as np
import cv2

from models import Page, Geometry, Word
from opencv_helpers import opencv_resize
from receipt_processing import get_doc_page_geometry, get_all_words_and_min_char_width, get_points, \
    get_doc_page_line_widths


class MergeParameters:
    """Helper Class for storing all the parameters involved in merging pages"""

    scale: float
    """factor to scale page_2 around the source point"""
    target_x: float
    """x position on page 1 that we want to move source position to"""
    target_y: float
    """y position on page 1 that we want to move source position to"""
    source_x: float
    """x position on page 2 that we want to move to target position"""
    source_y: float
    """y position on page 2 that we want to move to target position"""
    overlap_margin: float
    """amount of margin to ignore (larger values means geometries must overlap MORE to count as overlapping)"""

    overlap_ratio: int
    fuzz_ratio: int

    def __init__(self,
                 scale: float,
                 target_x: float,
                 target_y: float,
                 source_x: float,
                 source_y: float,
                 overlap_margin: float,
                 overlap_ratio: int = 0,
                 fuzz_ratio: int = 0,
                 ):
        self.scale = scale
        self.target_x = target_x
        self.target_y = target_y
        self.source_x = source_x
        self.source_y = source_y
        self.overlap_margin = overlap_margin
        self.overlap_ratio = overlap_ratio
        self.fuzz_ratio = fuzz_ratio

    def __str__(self):
        return f"overlap: {self.overlap_ratio}% | fuzz: {self.fuzz_ratio}% -> target_x={self.target_x}\t target_y={self.target_y}\t source_x={self.source_x}\t source_y={self.source_y}\t scale={self.scale}"

    def copy(self) -> MergeParameters:
        return MergeParameters(
            scale=self.scale,
            target_x=self.target_x,
            target_y=self.target_y,
            source_x=self.source_x,
            source_y=self.source_y,
            overlap_margin=self.overlap_margin,
            overlap_ratio=self.overlap_ratio,
            fuzz_ratio=self.fuzz_ratio,
        )


def calculate_overlap_geometry_fuzzy_match(
        page_1_geometry: Geometry,
        page_2_geometry: Geometry,
        page_1_words: list[Word],
        page_2_words: list[Word],
        merge_params: MergeParameters,
) -> (float, int):
    """
    helper function for calculating overlapping geometry in two pages of words
    :param page_1_geometry:
    :param page_2_geometry: page 2 geometry used for optimizing bounds checks by culling checks with pg1_words that are not overlapping page 2 at all
    :param page_1_words: list of words in page 1
    :param page_2_words: list of words in page 2
    :param merge_params: parameters used for calculating offsets & overlap
    :return: tuple, first value is ratio of overlapping words, second value is a rescaled fuzzy ratio
    """
    # for each overlapping pair of words, calculate the fuzzy ratio

    # calculate scaled deltas
    dx = merge_params.target_x - (merge_params.source_x * merge_params.scale)
    dy = merge_params.target_y - (merge_params.source_y * merge_params.scale)

    pg1_overlap_count = 0
    pg2_overlap_count = 0
    expected_pg1_words_with_overlap = 0
    fuzzy_ratio_sum = 0

    # pg1_x, pg1_y, pg1_x2, pg1_y2 = get_points(page_1_geometry)
    adjusted_pg2_geometry = Geometry(
        x=page_2_geometry.x * merge_params.scale + dx,
        y=page_2_geometry.y * merge_params.scale + dy,
        x2=page_2_geometry.x2 * merge_params.scale + dx,
        y2=page_2_geometry.y2 * merge_params.scale + dy,
    )

    for pg1_word in page_1_words:
        # pg1_word not overlapping adjusted pg2 bounds, so space computations and skip
        if not pg1_word.geometry.overlaps(adjusted_pg2_geometry):
            continue
        # sum for each pg1 word that overlaps pg2 geometry
        expected_pg1_words_with_overlap += 1

        was_overlap = False
        for pg2_word in page_2_words:
            # scale and offset the geometry
            adjusted_geometry = Geometry(
                x=pg2_word.geometry.x * merge_params.scale + dx,
                y=pg2_word.geometry.y * merge_params.scale + dy,
                x2=pg2_word.geometry.x2 * merge_params.scale + dx,
                y2=pg2_word.geometry.y2 * merge_params.scale + dy,
            )
            if pg1_word.geometry.overlaps(adjusted_geometry, merge_params.overlap_margin):
                # overlapping, increment pg2 overlap count
                was_overlap = True
                pg2_overlap_count += 1
                fuzzy_ratio_sum += fuzz.ratio(pg1_word.value, pg2_word.value)

        if was_overlap:
            pg1_overlap_count += 1

    if pg2_overlap_count == 0:
        print("no matches")
        return 0, 0

    expected_overlap_ratio = pg1_overlap_count / expected_pg1_words_with_overlap
    geometry_overlap_ratio = int((pg1_overlap_count / pg2_overlap_count) * expected_overlap_ratio * 100)
    rescaled_fuzzy = int((fuzzy_ratio_sum // pg2_overlap_count) * expected_overlap_ratio)
    print(f"{pg1_overlap_count}/{pg2_overlap_count}*{expected_overlap_ratio} = {geometry_overlap_ratio}% | {fuzzy_ratio_sum}/{pg2_overlap_count} = {rescaled_fuzzy}%")

    return geometry_overlap_ratio, rescaled_fuzzy


def get_merge_candidates(pg1_all_words: list[Word], pg2_all_words: list[Word]) -> dict[int, list[tuple[Word, int]]]:
    """find the most-likely word candidates to produce the best merge results

    :param pg1_all_words: page 1 words
    :param pg2_all_words: page 2 words
    :return: dict mapping from page 2 indices to a list of they're best matches (tuple of page 1 words and fuzz ratio)
    """
    # compile map of all the best matches from pg2 words to pg1 words based on fuzzy ratio
    best_matches: dict[
        int, list[tuple[Word, int]]] = {}  # index of pg2_all_words -> list of matched pg1_word & fuzzy ratio
    for i in range(len(pg2_all_words)):
        pg2_word = pg2_all_words[i]

        if len(pg2_word.value) <= 4:  # skip small words as alignment will be less accurate
            continue

        this_best_matches: list[tuple[Word, int]] = []
        for j in range(len(pg1_all_words) - 1, -1, -1):
            pg1_word = pg1_all_words[j]

            if len(pg1_word.value) <= 4:  # skip small words as alignment will be less accurate
                continue

            ratio = fuzz.ratio(pg1_word.value, pg2_word.value)
            # only find large ratio matches
            if ratio > 70:
                this_best_matches.append((pg1_word, ratio))

        if len(this_best_matches) > 0:
            best_matches[i] = this_best_matches

    # get the word that has the lowest y value
    top_word = sorted(list(best_matches.keys()), key=lambda i: pg2_all_words[i].geometry.y)[0]
    top_word_matches = best_matches.pop(top_word)

    # get the longest word
    longest_word = sorted(list(best_matches.keys()), key=lambda i: len(pg2_all_words[i].value), reverse=True)[0]
    longest_word_matches = best_matches.pop(longest_word)

    # get the word that has the fewest matches
    fewest_word = sorted(list(best_matches.keys()), key=lambda i: len(best_matches[i]))[0]
    fewest_word_matches = best_matches[fewest_word]

    # return candidate matches
    return {
        top_word: top_word_matches,
        longest_word: longest_word_matches,
        fewest_word: fewest_word_matches
    }


def merge_page_images(
        pg1_image,
        pg2_image,
        merge_params: MergeParameters,
        aggregate_pg2_image=None,
):
    if aggregate_pg2_image is None:
        aggregate_pg2_image = pg2_image

    # convert to pixels
    target_x = int(merge_params.target_x * pg1_image.shape[1])
    target_y = int(merge_params.target_y * pg1_image.shape[0])
    source_x = int(merge_params.source_x * pg2_image.shape[1])
    source_y = int(merge_params.source_y * pg2_image.shape[0])
    dx = target_x - int(source_x * merge_params.scale)
    dy = target_y - int(source_y * merge_params.scale)
    print(dx, dy)

    cp_pg2_img = aggregate_pg2_image.copy()
    cp_pg2_img = cv2.circle(cp_pg2_img, (source_x, source_y), 6, (0, 255, 0), -1)
    cp_pg2_img = opencv_resize(cp_pg2_img, merge_params.scale)

    print(pg1_image.shape)
    print(cp_pg2_img.shape)
    new_height = abs(dy) + cp_pg2_img.shape[0]
    new_width = max(pg1_image.shape[1], cp_pg2_img.shape[1])
    print(new_height, new_width)

    src_x = -dx if dx < 0 else 0
    src_y = -dy if dy < 0 else 0
    src_max_x = new_width - dx if dx > 0 else cp_pg2_img.shape[1]

    dest_x = dx if dx >= 0 else 0
    dest_y = dy if dy >= 0 else 0
    dest_max_x = dx + cp_pg2_img.shape[1]
    dest_max_y = dy + cp_pg2_img.shape[0]

    tmp = np.zeros(shape=[new_height, new_width, 3], dtype=np.uint8)
    tmp[:pg1_image.shape[0], :pg1_image.shape[1]] = pg1_image
    tmp[dest_y:dest_max_y, dest_x:dest_max_x] = cp_pg2_img[src_y:, src_x:src_max_x]
    tmp = cv2.circle(tmp, (target_x, target_y), 3, (0, 0, 255), -1)

    tmp = cv2.rectangle(tmp, (0, 0), (pg1_image.shape[1], pg1_image.shape[0]), (255, 255, 0), 2)

    return tmp


def get_merge_parameters(page_1: Page, page_2: Page) -> MergeParameters:
    # get page geometry as bounding box of all blocks in page
    pg1_geometry = get_doc_page_geometry(page_1, False)
    pg2_geometry = get_doc_page_geometry(page_2, False)

    pg1_all_words, pg1_char_width = get_all_words_and_min_char_width(page_1, False)
    pg2_all_words, pg2_char_width = get_all_words_and_min_char_width(page_2, False)
    avg_char_width_ratio = pg1_char_width / pg2_char_width
    print(f"{pg1_char_width} / {pg2_char_width} = {avg_char_width_ratio}")

    pg1_line_widths = get_doc_page_line_widths(page_1)
    pg2_line_widths = get_doc_page_line_widths(page_2)

    pg1_avg_line_width = sum(pg1_line_widths) / len(pg1_line_widths)
    pg2_avg_line_width = sum(pg2_line_widths) / len(pg2_line_widths)
    avg_line_width_ratio = pg1_avg_line_width / pg2_avg_line_width
    print(f"{pg1_avg_line_width} / {pg2_avg_line_width} = {avg_line_width_ratio}")

    pg1_median_line_width = sorted(pg1_line_widths)[len(pg1_line_widths) // 2]
    pg2_median_line_width = sorted(pg2_line_widths)[len(pg2_line_widths) // 2]
    median_line_width_ratio = pg1_median_line_width / pg2_median_line_width
    print(f"{pg1_median_line_width} / {pg2_median_line_width} = {median_line_width_ratio}")

    # compile candidate matches
    candidates = get_merge_candidates(pg1_all_words, pg2_all_words)

    # loop through candidates to find the best parameters
    best_merge_params = MergeParameters(
        scale=1,  # TODO
        target_x=-1,
        target_y=-1,
        source_x=-1,
        source_y=-1,
        overlap_margin=0.01,  # TODO
        overlap_ratio=0,
        fuzz_ratio=0,
    )
    for i in candidates:
        word = pg2_all_words[i]
        word_matches = candidates[i]

        word_char_width = (word.geometry.x2 - word.geometry.x) / len(word.value)
        for match in word_matches:
            (match_word, match_ratio) = match

            merge_params = MergeParameters(
                scale=1,  # TODO
                target_x=match_word.geometry.x,
                target_y=match_word.geometry.y,
                source_x=word.geometry.x,
                source_y=word.geometry.y,
                overlap_margin=0.01,  # TODO
            )

            match_char_width = (match_word.geometry.x2 - match_word.geometry.x) / len(match_word.value)
            char_width_ratio = match_char_width / word_char_width
            print()
            print(f"{word.value} -> {match_word.value} ... {char_width_ratio}")

            # attempt with a range of scales based on different ratio calculations
            min_scale = min(char_width_ratio, avg_char_width_ratio, median_line_width_ratio, avg_line_width_ratio) - 0.1
            max_scale = max(char_width_ratio, avg_char_width_ratio, median_line_width_ratio, avg_line_width_ratio) + 0.1
            num_steps = 4
            for j in range(num_steps+1):
                merge_params.scale = (max_scale - min_scale) * (j / num_steps) + min_scale

                # calculate overlap ratio
                overlap_ratio, fuzz_ratio = calculate_overlap_geometry_fuzzy_match(
                    pg1_geometry, pg2_geometry, pg1_all_words, pg2_all_words, merge_params)
                merge_params.overlap_ratio = overlap_ratio
                merge_params.fuzz_ratio = fuzz_ratio
                print(merge_params)

                # if merge_params.overlap_ratio > best_merge_params.overlap_ratio\
                #         or (merge_params.overlap_ratio == best_merge_params.overlap_ratio
                #             and merge_params.fuzz_ratio > best_merge_params.fuzz_ratio):
                if merge_params.overlap_ratio + merge_params.fuzz_ratio > best_merge_params.overlap_ratio + best_merge_params.fuzz_ratio:
                    # found new best!
                    best_merge_params = merge_params.copy()

    # TODO: more minor adjustments? or averaging of candidates... or use more candidates?

    return best_merge_params


def merge_all_pages(pages: list[Page],
                    page_paths: list[str] = None,
                    output_image_path: str = None) -> Page:
    merged_image = None
    for i in range(len(pages)-1, 0, -1):
        after_page = pages[i]
        before_page = pages[i-1]

        merge_params = get_merge_parameters(before_page, after_page)
        print("best merge params: ", merge_params)

        # TODO merge pages data

        if page_paths is not None and output_image_path is not None:
            after_page_img = cv2.imread(page_paths[i])
            before_page_img = cv2.imread(page_paths[i - 1])
            merged_image = merge_page_images(before_page_img, after_page_img, merge_params, merged_image)

    if merged_image is not None:
        cv2.imwrite(output_image_path, merged_image)

    result_page = dict # TODO
    return result_page