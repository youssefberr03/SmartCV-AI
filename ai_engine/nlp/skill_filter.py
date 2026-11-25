
from .skill_repository import TECHNICAL_SKILLS


class SkillFilter:

    GENERIC_TERMS = {
        "current", "equipment", "data",
        "reporting", "engineering", "processes",
        "process", "security", "board",
        "standard", "standards", "tie",
        "pole", "press", "packing",
        "power", "work", "use", "make",
        "plan", "planning", "support",
        "service", "services", "system",
        "systems", "information", "business",
        "operations", "operation", "development",
        "management", "design", "access",
        "office", "spring", "go", "r", "c",
        "experience", "knowledge", "skills",
        "team", "project", "projects",
        "company", "product", "products",
        "production", "general", "technical",

        "courant", "équipement", "données",
        "sécurité", "processus", "gestion",
        "développement", "travail", "service",
        "entreprise", "expérience", "projet",
        "information", "système", "équipe",
        "connaissance", "produit"
    }

    WEAK_START_WORDS = {
        "and", "or", "of", "the",
        "with", "for", "in", "to",
        "a", "an", "by", "from",

        "et", "ou", "de", "du",
        "des", "avec", "pour", "dans",
        "le", "la", "les", "un", "une"
    }

    BLOCKED_ALIASES = {
        "electricity principles": {
            "current",
            "voltage",
            "resistance"
        },
        "compile airport certification manuals": {
            "equipment",
            "and procedures"
        },
        "journalism": {
            "reporting",
            "press"
        },
        "packaging engineering": {
            "engineering",
            "packing"
        },
        "perform ground-handling maintenance procedures": {
            "processes"
        },
        "scaffolding components": {
            "board",
            "standard",
            "tie",
            "pole"
        },
        "securities": {
            "security"
        },
        "statistics": {
            "data",
            "data analysis",
            "use of data"
        }
    }

    TRUSTED_TECHNICAL_TERMS = {
        alias.casefold()
        for aliases in TECHNICAL_SKILLS.values()
        for alias in aliases
    }

    def normalize(self, text):
        return " ".join(
            text.casefold().split()
        )

    def get_rejection_reason(self, alias, skill):
        normalized = self.normalize(alias)

        if len(normalized) < 2:
            return "too_short"

        if not any(
            character.isalpha()
            for character in normalized
        ):
            return "no_alphabetic_character"

        labels = {
            self.normalize(label)
            for label in skill["labels"].values()
        }

        for label in labels:
            blocked = self.BLOCKED_ALIASES.get(
                label,
                set()
            )

            if normalized in blocked:
                return "blocked_concept_alias"

        if normalized in self.GENERIC_TERMS:
            return "generic_term"

        words = normalized.split()

        if words[0] in self.WEAK_START_WORDS:
            return "weak_expression"

        if normalized in self.TRUSTED_TECHNICAL_TERMS:
            return None

        if len(words) == 1:
            if normalized not in labels:
                return "unverified_single_word_alias"

        return None

    def is_valid(self, alias, skill):
        return (
            self.get_rejection_reason(alias, skill)
            is None
        )
