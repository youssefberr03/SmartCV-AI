
from pathlib import Path

import pymupdf

from .extraction_models import (
    TextBlock,
    PageExtraction,
    PDFExtractionResult
)


class PDFExtractor:

    MAX_FILE_SIZE = 20 * 1024 * 1024
    MAX_PAGES = 30
    MIN_TEXT_CHARACTERS = 30

    def validate_file(self, file_path):
        path = Path(file_path)

        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        if path.suffix.lower() != ".pdf":
            raise ValueError("Only PDF files are supported")

        if path.stat().st_size == 0:
            raise ValueError("The PDF file is empty")

        if path.stat().st_size > self.MAX_FILE_SIZE:
            raise ValueError("The PDF exceeds the size limit")

        return path

    def extract_blocks(self, page):
        blocks = []

        for block in page.get_text("blocks", sort=True):
            x0, y0, x1, y1, text, _, block_type = block

            if block_type != 0 or not text.strip():
                continue

            blocks.append(
                TextBlock(
                    text=text.strip(),
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1
                )
            )

        return blocks

    def extract_page(self, page, page_number):
        text = page.get_text("text", sort=True).strip()

        blocks = self.extract_blocks(page)
        image_count = len(page.get_image_info())

        requires_ocr = (
            len(text) < self.MIN_TEXT_CHARACTERS
            and image_count > 0
        )

        return PageExtraction(
            page_number=page_number,
            text=text,
            character_count=len(text),
            word_count=len(text.split()),
            image_count=image_count,
            requires_ocr=requires_ocr,
            blocks=blocks
        )

    def extract(self, file_path):
        path = self.validate_file(file_path)

        pages = []
        warnings = []

        with pymupdf.open(str(path)) as document:

            if not document.is_pdf:
                raise ValueError("The file is not a valid PDF")

            if document.needs_pass:
                raise ValueError("Password-protected PDF")

            if document.page_count == 0:
                raise ValueError("The PDF contains no pages")

            if document.page_count > self.MAX_PAGES:
                raise ValueError("The PDF exceeds the page limit")

            for index, page in enumerate(document):
                result = self.extract_page(
                    page,
                    page_number=index + 1
                )

                pages.append(result)

                if result.requires_ocr:
                    warnings.append(
                        f"Page {index + 1}: OCR may be required"
                    )

                elif not result.text:
                    warnings.append(
                        f"Page {index + 1}: No text detected"
                    )

        extracted_text = "\n\n".join(
            page.text for page in pages if page.text
        )

        if not extracted_text:
            warnings.append("No extractable text found")

        return PDFExtractionResult(
            file_name=path.name,
            file_size_bytes=path.stat().st_size,
            page_count=len(pages),
            character_count=len(extracted_text),
            word_count=len(extracted_text.split()),
            extracted_text=extracted_text,
            requires_ocr=any(
                page.requires_ocr for page in pages
            ),
            pages=pages,
            warnings=warnings
        )
