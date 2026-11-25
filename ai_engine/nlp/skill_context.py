
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ContextDecision:
    status: str
    reason: str


class SkillContextAnalyzer:

    MAX_CONTEXT_LENGTH = 200

    BOUNDARY_PATTERN = re.compile(
        r"[.!?;](?=\s|$)|[\r\n]"
        r"|\b(?:but|however|although|whereas"
        r"|mais|cependant|toutefois|pourtant)\b",
        re.IGNORECASE
    )

    NEGATION_PATTERN = re.compile(
        r"\b(?:"
        r"no\s+(?:prior\s+)?experience\s+(?:with|in|of)"
        r"|not\s+familiar\s+with"
        r"|without\s+experience\s+(?:with|in)"
        r"|never\s+(?:used|worked\s+with)"
        r"|do\s+not\s+know"
        r"|don['’]t\s+know"
        r"|aucune\s+expérience\s+(?:en|avec)"
        r"|pas\s+d['’]expérience\s+(?:en|avec)"
        r"|ne\s+maîtrise\s+pas"
        r"|jamais\s+utilisé"
        r")\s*",
        re.IGNORECASE
    )

    LEARNING_PATTERN = re.compile(
        r"\b(?:"
        r"(?:currently\s+)?learning"
        r"|beginner\s+in"
        r"|training\s+in"
        r"|studying"
        r"|getting\s+started\s+with"
        r"|en\s+cours\s+d['’]apprentissage\s+(?:de|du)"
        r"|en\s+formation\s+(?:en|sur)"
        r"|j['’]apprends"
        r"|débutant\s+en"
        r")\s*",
        re.IGNORECASE
    )

    INTEREST_PATTERN = re.compile(
        r"\b(?:"
        r"interested\s+in"
        r"|interest\s+in"
        r"|passionate\s+about"
        r"|curious\s+about"
        r"|intéressé(?:e)?\s+par"
        r"|intérêt\s+pour"
        r"|passionné(?:e)?\s+par"
        r")\s*",
        re.IGNORECASE
    )

    ASSERTION_PATTERN = re.compile(
        r"\b(?:"
        r"(?:prior\s+)?experience\s+(?:with|in)"
        r"|experienced\s+in"
        r"|proficient\s+in"
        r"|skilled\s+in"
        r"|expertise\s+in"
        r"|knowledge\s+of"
        r"|familiar\s+with"
        r"|worked\s+with"
        r"|used"
        r"|degree\s+in"
        r"|native\s+language"
        r"|technical\s+skills"
        r"|skills"
        r"|expérience\s+(?:en|avec)"
        r"|compétences\s+(?:en|techniques)"
        r"|connaissance\s+(?:de|du)"
        r"|maîtrise\s+(?:de|du)"
        r"|je\s+maîtrise"
        r"|langue\s+maternelle"
        r"|diplôme\s+en"
        r")\s*",
        re.IGNORECASE
    )

    CONTINUATION_PATTERN = re.compile(
        r"^(?:"
        r"\s"
        r"|[,/:&()]"
        r"|<SKILL>"
        r"|\b(?:and|or|nor|et|ou|ni|either)\b"
        r"|\b(?:le|la|les|un|une|des)\b"
        r"|\b[ld]['’]"
        r")*$",
        re.IGNORECASE
    )

    HISTORY_PREFIX = re.compile(
        r"\b(?:employment|work|career|professional"
        r"|medical|job|company|credit|purchase"
        r"|browsing|patient)\s*$",
        re.IGNORECASE
    )

    HISTORY_SUFFIX = re.compile(
        r"^\s+(?:of\s+employment|of\s+work"
        r"|professionnel|médical)\b",
        re.IGNORECASE
    )

    RULES = (
        ("negated", NEGATION_PATTERN),
        ("learning", LEARNING_PATTERN),
        ("interest", INTEREST_PATTERN),
        ("asserted", ASSERTION_PATTERN)
    )

    def get_prefix(self, text, start_char):
        start = max(
            0,
            start_char - self.MAX_CONTEXT_LENGTH
        )

        prefix = text[start:start_char]

        boundaries = list(
            self.BOUNDARY_PATTERN.finditer(prefix)
        )

        if boundaries:
            boundary = boundaries[-1].end()
            start += boundary
            prefix = prefix[boundary:]

        return prefix, start

    def is_false_positive(
        self,
        text,
        start_char,
        end_char,
        skill
    ):
        mention = text[
            start_char:end_char
        ].casefold()

        labels = {
            label.casefold()
            for label in skill["labels"].values()
        }

        if (
            mention == "history"
            and "history" in labels
        ):
            prefix = text[
                max(0, start_char - 60):start_char
            ]

            suffix = text[
                end_char:end_char + 60
            ]

            if self.HISTORY_PREFIX.search(prefix):
                return True

            if self.HISTORY_SUFFIX.search(suffix):
                return True

        return False

    def mask_previous_mentions(
        self,
        prefix,
        prefix_start,
        previous_mentions
    ):
        prefix_end = prefix_start + len(prefix)

        relevant_mentions = [
            (start, end)
            for start, end in previous_mentions
            if start >= prefix_start
            and end <= prefix_end
        ]

        for start, end in sorted(
            relevant_mentions,
            reverse=True
        ):
            local_start = start - prefix_start
            local_end = end - prefix_start

            prefix = (
                prefix[:local_start]
                + " <SKILL> "
                + prefix[local_end:]
            )

        return prefix

    def classify(
        self,
        text,
        start_char,
        end_char,
        skill,
        previous_mentions=()
    ):
        if self.is_false_positive(
            text,
            start_char,
            end_char,
            skill
        ):
            return ContextDecision(
                status="rejected",
                reason="contextual_false_positive"
            )

        prefix, prefix_start = self.get_prefix(
            text,
            start_char
        )

        prefix = self.mask_previous_mentions(
            prefix,
            prefix_start,
            previous_mentions
        )

        candidates = []

        for priority, (status, pattern) in enumerate(
            self.RULES
        ):
            for match in pattern.finditer(prefix):
                continuation = prefix[match.end():]

                if not self.CONTINUATION_PATTERN.fullmatch(
                    continuation
                ):
                    continue

                candidates.append((
                    match.end(),
                    len(match.group()),
                    -priority,
                    status
                ))

        if candidates:
            _, _, _, status = max(candidates)

            return ContextDecision(
                status=status,
                reason=f"context_rule_{status}"
            )

        return ContextDecision(
            status="mentioned",
            reason="lexical_match_only"
        )
