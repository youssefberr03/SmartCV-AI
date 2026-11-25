
from collections import defaultdict, Counter

import spacy
from spacy.matcher import PhraseMatcher

from .skill_repository import SkillRepository
from .skill_filter import SkillFilter
from .skill_context import SkillContextAnalyzer
from .technical_tokenizer import configure_technical_tokenizer


class SkillExtractor:

    POSITIVE_STATUSES = {
        "asserted",
        "mentioned"
    }

    def __init__(self, repository=None):
        if repository is None:
            repository = SkillRepository().load()

        self.repository = repository

        self.pipelines = {}
        self.matchers = {}

        self.skill_filter = SkillFilter()
        self.context_analyzer = SkillContextAnalyzer()

        self.filter_statistics = {}

    def build_matcher(self, language):
        if language not in {"en", "fr"}:
            raise ValueError(
                f"Unsupported language: {language}"
            )

        # Configure language-specific technical tokenization.
        nlp = configure_technical_tokenizer(
            spacy.blank(language)
        )

        matcher = PhraseMatcher(
            nlp.vocab,
            attr="LOWER"
        )

        alias_index = defaultdict(set)
        statistics = Counter()

        for skill_id, skill in self.repository.items():

            aliases = set(
                skill["aliases"].get(language, [])
            )

            other_language = (
                "fr" if language == "en" else "en"
            )

            aliases.update(
                skill["aliases"].get(other_language, [])
            )

            for alias in aliases:

                statistics[
                    "total_alias_candidates"
                ] += 1

                rejection_reason = (
                    self.skill_filter.get_rejection_reason(
                        alias,
                        skill
                    )
                )

                if rejection_reason is not None:

                    statistics[
                        f"rejected_{rejection_reason}"
                    ] += 1

                    continue

                doc = nlp.make_doc(alias)

                key = tuple(
                    token.lower_
                    for token in doc
                )

                if not key:

                    statistics[
                        "rejected_empty_pattern"
                    ] += 1

                    continue

                alias_index[key].add(skill_id)

                statistics[
                    "accepted_alias_candidates"
                ] += 1

        patterns_by_skill = defaultdict(list)

        # Exclude patterns associated with multiple concepts.
        for tokens, skill_ids in alias_index.items():

            if len(skill_ids) != 1:

                statistics[
                    "ambiguous_patterns"
                ] += 1

                continue

            skill_id = next(iter(skill_ids))

            patterns_by_skill[skill_id].append(
                spacy.tokens.Doc(
                    nlp.vocab,
                    words=list(tokens)
                )
            )

            statistics[
                "registered_patterns"
            ] += 1

        for skill_id, patterns in patterns_by_skill.items():

            matcher.add(
                skill_id,
                patterns
            )

        statistics["registered_concepts"] = len(
            patterns_by_skill
        )

        self.filter_statistics[language] = dict(
            statistics
        )

        self.pipelines[language] = nlp
        self.matchers[language] = matcher

        return nlp, matcher

    def get_matcher(self, language):

        if language not in self.matchers:
            return self.build_matcher(language)

        return (
            self.pipelines[language],
            self.matchers[language]
        )

    def get_filter_statistics(self, language="en"):

        self.get_matcher(language)

        return self.filter_statistics[language]

    def resolve_overlaps(self, matches):

        # Prioritize the longest matching expressions.
        ordered = sorted(
            matches,
            key=lambda match: (
                -(match[2] - match[1]),
                match[1],
                match[0]
            )
        )

        selected = []
        occupied_tokens = set()

        for match_id, start, end in ordered:

            positions = set(
                range(start, end)
            )

            if positions & occupied_tokens:
                continue

            selected.append(
                (match_id, start, end)
            )

            occupied_tokens.update(positions)

        return sorted(
            selected,
            key=lambda match: match[1]
        )

    def create_skill_entry(
        self,
        skill_id,
        skill,
        language
    ):

        return {
            "id": skill_id,

            "label": skill["labels"].get(
                language,
                next(iter(skill["labels"].values()))
            ),

            "source": skill["source"],

            "positive_mentions": 0,
            "asserted_mentions": 0,
            "mentioned_mentions": 0,
            "negative_mentions": 0,
            "learning_mentions": 0,
            "interest_mentions": 0,

            "mentions": []
        }

    def update_skill_entry(
        self,
        entry,
        span,
        decision
    ):

        status = decision.status

        entry["mentions"].append({
            "text": span.text,
            "start_char": span.start_char,
            "end_char": span.end_char,
            "status": status,
            "reason": decision.reason,
            "negated": status == "negated"
        })

        if status in self.POSITIVE_STATUSES:

            entry["positive_mentions"] += 1

        counter_mapping = {
            "asserted": "asserted_mentions",
            "mentioned": "mentioned_mentions",
            "negated": "negative_mentions",
            "learning": "learning_mentions",
            "interest": "interest_mentions"
        }

        counter = counter_mapping.get(status)

        if counter:
            entry[counter] += 1

    def extract(self, text, language="en"):

        if not isinstance(text, str):

            raise TypeError(
                "Input must be a string"
            )

        nlp, matcher = self.get_matcher(language)

        doc = nlp.make_doc(text)

        matches = self.resolve_overlaps(
            matcher(doc)
        )

        detected = {}
        rejected_mentions = []
        previous_mentions = []

        context_distribution = Counter()

        for match_id, start, end in matches:

            skill_id = nlp.vocab.strings[match_id]

            skill = self.repository[skill_id]

            span = doc[start:end]

            decision = self.context_analyzer.classify(
                text=text,
                start_char=span.start_char,
                end_char=span.end_char,
                skill=skill,
                previous_mentions=previous_mentions
            )

            context_distribution[
                decision.status
            ] += 1

            if decision.status == "rejected":

                rejected_mentions.append({

                    "id": skill_id,

                    "label": skill["labels"].get(
                        language,
                        next(iter(
                            skill["labels"].values()
                        ))
                    ),

                    "text": span.text,

                    "start_char": span.start_char,

                    "end_char": span.end_char,

                    "reason": decision.reason
                })

                continue

            if skill_id not in detected:

                detected[skill_id] = (
                    self.create_skill_entry(
                        skill_id,
                        skill,
                        language
                    )
                )

            self.update_skill_entry(
                detected[skill_id],
                span,
                decision
            )

            previous_mentions.append((
                span.start_char,
                span.end_char
            ))

        skills = []
        negated_skills = []
        learning_skills = []
        interest_skills = []
        conflicting_skills = []

        # Aggregate individual mentions at the concept level.
        for skill in detected.values():

            has_positive = (
                skill["positive_mentions"] > 0
            )

            has_negative = (
                skill["negative_mentions"] > 0
            )

            if has_positive and has_negative:

                conflicting_skills.append(skill)

            elif has_positive:

                skills.append(skill)

            elif has_negative:

                negated_skills.append(skill)

            elif skill["learning_mentions"] > 0:

                learning_skills.append(skill)

            elif skill["interest_mentions"] > 0:

                interest_skills.append(skill)

        return {
            "language": language,

            "skill_count": len(skills),

            "negated_skill_count": len(
                negated_skills
            ),

            "learning_skill_count": len(
                learning_skills
            ),

            "interest_skill_count": len(
                interest_skills
            ),

            "conflicting_skill_count": len(
                conflicting_skills
            ),

            "rejected_mention_count": len(
                rejected_mentions
            ),

            "skills": skills,

            "negated_skills": negated_skills,

            "learning_skills": learning_skills,

            "interest_skills": interest_skills,

            "conflicting_skills": conflicting_skills,

            "rejected_mentions": rejected_mentions,

            "context_distribution": dict(
                context_distribution
            )
        }
