
from spacy.symbols import ORTH


TECHNICAL_TERMS = (
    "C++",
    "C#",
    ".NET",
    "Node.js",
    "React.js",
    "Vue.js",
    "Next.js",
    "CI/CD",
    "ASP.NET"
)

TRAILING_PUNCTUATION = (
    ".",
    ",",
    ";",
    ":",
    "!",
    "?"
)


def configure_technical_tokenizer(nlp):
    for term in TECHNICAL_TERMS:
        variants = {
            term,
            term.lower(),
            term.upper()
        }

        for variant in variants:
            nlp.tokenizer.add_special_case(
                variant,
                [{ORTH: variant}]
            )

            for punctuation in TRAILING_PUNCTUATION:
                nlp.tokenizer.add_special_case(
                    variant + punctuation,
                    [
                        {ORTH: variant},
                        {ORTH: punctuation}
                    ]
                )

    return nlp
