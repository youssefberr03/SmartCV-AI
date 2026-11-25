from dataclasses import dataclass ,asdict,field

@dataclass
class TextBlock:
    text: str
    x0 : float
    y0 : float
    x1 : float
    y1 : float

@dataclass
class PageExtraction:
    page_number: int
    text: str
    character_count: int
    word_count: int
    image_count: int
    requires_ocr: bool
    blocks: list[TextBlock] = field(default_factory=list)


@dataclass
class PDFExtractionResult:
    file_name: str
    file_size_bytes: int
    page_count: int
    character_count: int
    word_count: int
    extracted_text: str
    requires_ocr: bool
    pages: list[PageExtraction] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)
