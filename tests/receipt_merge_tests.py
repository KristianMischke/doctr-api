import os
import unittest

from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from doctr.models.predictor import OCRPredictor

from receipt_merge import calculate_overlap_geometry_fuzzy_match, MergeParameters, merge_page_images, merge_all_pages
from receipt_processing import *


class ReceiptMergeTests(unittest.TestCase):
    model: OCRPredictor
    img_tests: dict[str, dict]

    @classmethod
    def get_test_images(cls):
        # collect images in data dir by index and store parts in array
        cls.img_tests: dict[str, dict] = {}
        img_filenames = sorted(list(os.listdir("data")))
        for img_filename in img_filenames:
            name_parts = img_filename.split("_")
            idx = name_parts[0]
            label = name_parts[2]

            fullpath = os.path.join("data", img_filename)
            if idx in cls.img_tests:
                cls.img_tests[idx]["file_parts"].append(fullpath)
            else:
                cls.img_tests[idx] = {
                    "label": label,
                    "file_parts": [fullpath]
                }
        print(cls.img_tests)

        # loop over each index to test
        for idx in cls.img_tests:
            img_test = cls.img_tests[idx]

            # get doc pages as list
            page_parts = []
            for part_filepath in img_test["file_parts"]:
                doc = DocumentFile.from_images(part_filepath)
                result = cls.model(doc)
                page_parts.append(Page.get_from_doctr_page(result.pages[0]))
            img_test["page_parts"] = page_parts

    @classmethod
    def setUpClass(cls):
        cls.model: OCRPredictor = ocr_predictor(pretrained=True).cuda()
        cls.get_test_images()

    def test_calculate_overlap_self(self):
        for idx in ReceiptMergeTests.img_tests:
            img_test = ReceiptMergeTests.img_tests[idx]

            page_i = 0
            for page in img_test["page_parts"]:
                pg1_all_words, pg1_char_width = get_all_words_and_min_char_width(page, False)
                pg1_geometry = get_doc_page_geometry(page, False)

                overlap_margin = 0.01
                overlap_ratio, fuzz_ratio = calculate_overlap_geometry_fuzzy_match(
                    pg1_geometry, pg1_geometry, pg1_all_words, pg1_all_words, MergeParameters(1, 0, 0, 0, 0, 0.006))
                self.assertLessEqual(95, overlap_ratio, f"idx {idx} page_i {page_i}")
                self.assertLessEqual(95, fuzz_ratio, f"idx {idx} page_i {page_i}")

                overlap_ratio, fuzz_ratio = calculate_overlap_geometry_fuzzy_match(
                    pg1_geometry, pg1_geometry, pg1_all_words, pg1_all_words, MergeParameters(1, 0, 0, 0, 0, overlap_margin))
                self.assertLessEqual(95, overlap_ratio, f"idx {idx} page_i {page_i}")
                self.assertLessEqual(95, fuzz_ratio, f"idx {idx} page_i {page_i}")

                overlap_ratio, fuzz_ratio = calculate_overlap_geometry_fuzzy_match(
                    pg1_geometry, pg1_geometry, pg1_all_words, pg1_all_words, MergeParameters(2, 0.3, 0, 0, 0, overlap_margin))
                self.assertGreater(100, overlap_ratio, f"idx {idx} page_i {page_i}")
                self.assertGreater(100, fuzz_ratio, f"idx {idx} page_i {page_i}")

                page_i += 1

            print()

    def test_receipt_merge(self):
        # loop over each index to test
        for idx in ReceiptMergeTests.img_tests:
            # if idx != '007':
            #     continue
            img_test = ReceiptMergeTests.img_tests[idx]
            label = img_test["label"]
            page_parts = img_test["page_parts"]
            file_parts = img_test["file_parts"]

            print(f"\n\n---merging {idx}---\n")
            merge_all_pages(page_parts, file_parts, f"./output/{idx}_merge.jpg")

            # merge pages and test
            # result = merge_doc_pages(page_parts)
            # result_text = convert_doc_page_to_text_grid(result, False)
            # self.assertEqual("", result_text)


if __name__ == '__main__':
    unittest.main()
