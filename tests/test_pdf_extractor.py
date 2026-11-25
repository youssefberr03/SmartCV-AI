
import pymupdf
import pytest

from ai_engine.extraction.pdf_extractor import PDFExtractor


@pytest.fixture
def sample_pdf(tmp_path):
    path = tmp_path / "sample_cv.pdf"

    with pymupdf.open() as document:
        page = document.new_page()

        page.insert_text(
            (50, 50),
            "Software Developer"
        )

        page.insert_text(
            (50, 80),
            "Skills: Java, Python, PostgreSQL"
        )

        document.save(str(path))

    return path


def test_valid_pdf(sample_pdf):
    extractor = PDFExtractor()
    result = extractor.extract(sample_pdf)

    assert result.page_count == 1
    assert result.word_count > 0
    assert result.character_count > 0


def test_extracted_text(sample_pdf):
    result = PDFExtractor().extract(sample_pdf)

    assert "Software Developer" in result.extracted_text
    assert "Python" in result.extracted_text
    assert "PostgreSQL" in result.extracted_text


def test_text_blocks(sample_pdf):
    result = PDFExtractor().extract(sample_pdf)

    assert len(result.pages[0].blocks) > 0

    for block in result.pages[0].blocks:
        assert block.x0 < block.x1
        assert block.y0 < block.y1


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        PDFExtractor().extract(
            tmp_path / "missing.pdf"
        )


def test_invalid_extension(tmp_path):
    path = tmp_path / "document.txt"
    path.write_text("Example text")

    with pytest.raises(ValueError):
        PDFExtractor().extract(path)


def test_empty_file(tmp_path):
    path = tmp_path / "empty.pdf"
    path.touch()

    with pytest.raises(ValueError):
        PDFExtractor().extract(path)


def test_no_ocr_required(sample_pdf):
    result = PDFExtractor().extract(sample_pdf)

    assert result.requires_ocr is False


def test_blank_pdf(tmp_path):
    path = tmp_path / "blank.pdf"

    with pymupdf.open() as document:
        document.new_page()
        document.save(str(path))

    result = PDFExtractor().extract(path)

    assert result.extracted_text == ""
    assert result.word_count == 0
    assert len(result.warnings) > 0


def test_serialization(sample_pdf):
    result = PDFExtractor().extract(sample_pdf)

    data = result.to_dict()

    assert isinstance(data, dict)
    assert isinstance(data["pages"], list)
    assert data["page_count"] == 1
