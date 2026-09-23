"""
Synthetic English Data Generator for ScratchLM Phase 1.

Generates targeted, high-quality synthetic English passages (~20% of corpus)
designed to fill gaps in natural text datasets:
- Grammar examples & sentence transformations
- Vocabulary & synonym contrasts
- Multi-turn conversations & Q&A
- Multi-level writing (elementary, intermediate, advanced)
- Grammar mistake corrections
- Punctuation & rhetorical exercises
- Varied sentence structures
"""

import random
from typing import List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class SyntheticDocument:
    """A synthetically generated English text document with provenance metadata."""
    doc_id: str
    text: str
    category: str
    subcategory: str
    difficulty: str
    source_name: str = "ScratchLM Synthetic English Generator v1.0"
    license: str = "CC0 1.0 Universal (Public Domain Dedication)"
    is_synthetic: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class SyntheticEnglishGenerator:
    """Generates varied synthetic English documents across multiple linguistic dimensions."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_all(self, target_count: int = 250) -> List[SyntheticDocument]:
        """Generate a total pool of synthetic documents distributed across categories."""
        docs: List[SyntheticDocument] = []
        generators = [
            (self.generate_grammar_transformations, int(target_count * 0.15)),
            (self.generate_word_contrasts, int(target_count * 0.15)),
            (self.generate_qa_pairs, int(target_count * 0.15)),
            (self.generate_multi_level_explanations, int(target_count * 0.15)),
            (self.generate_grammar_corrections, int(target_count * 0.15)),
            (self.generate_conversations, int(target_count * 0.15)),
            (self.generate_punctuation_and_rhetoric, target_count - 6 * int(target_count * 0.15)),
        ]

        doc_idx = 1
        for gen_fn, count in generators:
            for _ in range(count):
                doc = gen_fn(f"synth_{doc_idx:05d}")
                docs.append(doc)
                doc_idx += 1

        self.rng.shuffle(docs)
        return docs

    def generate_grammar_transformations(self, doc_id: str) -> SyntheticDocument:
        """Active/passive, direct/indirect speech, conditional transformations."""
        templates = [
            ("Active to Passive Voice", [
                "Active Voice: The engineer designed a durable bridge across the river.",
                "Passive Voice: A durable bridge across the river was designed by the engineer.",
                "Active Voice: Scientists discovered a new species of deep-sea organism.",
                "Passive Voice: A new species of deep-sea organism was discovered by scientists.",
                "Explanation: In passive voice, the object of the active sentence becomes the subject, emphasizing the receiver of the action rather than the performer."
            ]),
            ("Conditionals (First, Second, Third)", [
                "First Conditional (Real Future): If the temperature drops below freezing tonight, the water in the pipe will expand.",
                "Second Conditional (Unreal Present): If I had unlimited resources, I would explore the deepest trenches of the ocean.",
                "Third Conditional (Unreal Past): If the team had checked the equipment beforehand, they would have noticed the calibration error.",
                "Grammar Note: Conditionals express hypothetical scenarios using if-clauses paired with modal verbs like will, would, or could."
            ]),
            ("Relative Clauses", [
                "Simple Sentence 1: The astronomer published a groundbreaking paper on exoplanets.",
                "Simple Sentence 2: She won a prestigious award last month.",
                "Combined with Relative Clause: The astronomer who published a groundbreaking paper on exoplanets won a prestigious award last month.",
                "Non-defining Relative Clause: Exoplanets, which are planets orbiting stars outside our solar system, offer valuable clues about planetary formation."
            ]),
        ]

        title, lines = self.rng.choice(templates)
        text = f"Linguistic Topic: {title}\n\n" + "\n".join(lines)
        return SyntheticDocument(
            doc_id=doc_id,
            text=text,
            category="grammar_and_syntax",
            subcategory="sentence_transformations",
            difficulty="intermediate",
            metadata={"topic": title, "type": "structural_grammar"}
        )

    def generate_word_contrasts(self, doc_id: str) -> SyntheticDocument:
        """Differentiating commonly confused words and synonyms in context."""
        pairs = [
            ("Affect vs. Effect", [
                "Affect is primarily a verb meaning 'to influence or produce a change in something.'",
                "Example: How will the sudden shift in weather affect the crop yield this season?",
                "Effect is primarily a noun meaning 'the result or outcome of a cause.'",
                "Example: The positive effect of regular physical exercise on mental health is well-documented.",
                "Exception: Effect can function as a verb meaning 'to bring about' (e.g., 'effecting meaningful reform')."
            ]),
            ("Principal vs. Principle", [
                "Principal can be an adjective meaning 'chief or primary' or a noun meaning 'head of a school' or 'capital sum.'",
                "Example: The principal reason for the meeting was to address budget constraints.",
                "Principle is a noun meaning 'a fundamental truth, rule, or belief.'",
                "Example: She refused to compromise on her core ethical principles."
            ]),
            ("Continuous vs. Continual", [
                "Continuous means occurring without any interruption or break in time.",
                "Example: The steady, continuous hum of the machine indicated that it was operating smoothly.",
                "Continual means happening repeatedly over time with brief interruptions.",
                "Example: His speech was interrupted by continual applause from the enthusiastic audience."
            ])
        ]

        word_pair, lines = self.rng.choice(pairs)
        text = f"Vocabulary Comparison: {word_pair}\n\n" + "\n".join(lines)
        return SyntheticDocument(
            doc_id=doc_id,
            text=text,
            category="vocabulary_and_semantics",
            subcategory="word_contrasts",
            difficulty="intermediate",
            metadata={"pair": word_pair, "type": "vocabulary_precision"}
        )

    def generate_qa_pairs(self, doc_id: str) -> SyntheticDocument:
        """Structured Question and Answer sets across science, geography, and general knowledge."""
        qa_sets = [
            [
                ("Question: How does photosynthesis work in green plants?",
                 "Answer: Photosynthesis is the process by which green plants, algae, and certain bacteria convert light energy into chemical energy. Using chlorophyll, plants absorb sunlight, water from the soil, and carbon dioxide from the air. Through a series of chemical reactions, they produce glucose for nourishment and release oxygen as a byproduct."),
                ("Question: Why is photosynthesis important for Earth's ecosystem?",
                 "Answer: Photosynthesis supplies the primary source of organic matter for almost all life on Earth and maintains atmospheric oxygen levels necessary for aerobic organisms to breathe.")
            ],
            [
                ("Question: What causes the phenomenon known as ocean tides?",
                 "Answer: Ocean tides are caused primarily by the gravitational pull exerted by the Moon and the Sun on Earth's oceans. As Earth rotates, the differential gravitational forces create tidal bulges on opposite sides of the planet, leading to high and low tides twice daily in most coastal locations."),
                ("Question: What is the difference between spring tides and neap tides?",
                 "Answer: Spring tides occur when the Earth, Moon, and Sun align during full and new moons, producing maximum tidal range. Neap tides occur during quarter moons when the Sun and Moon pull at right angles, resulting in minimal tidal variation.")
            ]
        ]

        qa_list = self.rng.choice(qa_sets)
        formatted = "\n\n".join([f"{q}\n{a}" for q, a in qa_list])
        return SyntheticDocument(
            doc_id=doc_id,
            text=formatted,
            category="educational_qa",
            subcategory="explanations",
            difficulty="intermediate",
            metadata={"num_qa": len(qa_list), "type": "structured_qa"}
        )

    def generate_multi_level_explanations(self, doc_id: str) -> SyntheticDocument:
        """Explaining a complex idea at beginner, intermediate, and advanced English levels."""
        topics = [
            ("Gravity", [
                "Beginner Level: Gravity is the invisible force that pulls things toward each other. It keeps our feet on the ground and causes apples to drop from trees.",
                "Intermediate Level: Gravity is a fundamental natural force where objects with mass attract one another. The strength of gravitational attraction depends on two factors: the mass of the objects and the distance separating them.",
                "Advanced Level: According to Albert Einstein's theory of general relativity, gravity is not merely a mechanical pulling force, but rather a curvature of spacetime caused by the presence of mass and energy. Massive objects warp the geometry of space around them, guiding the trajectory of nearby matter and light."
            ]),
            ("Biodiversity", [
                "Beginner Level: Biodiversity means having many different kinds of plants, animals, and insects living in a forest or ocean.",
                "Intermediate Level: Biodiversity refers to the variety of life on Earth at all levels, from genetic variation within species to the diversity of ecosystems across landscapes.",
                "Advanced Level: Ecosystem resilience relies fundamentally on high genetic and species diversity, which enhances adaptive capacity in response to environmental disturbances, climate shifts, and anthropogenic pressures."
            ])
        ]

        title, levels = self.rng.choice(topics)
        text = f"Concept Explanation at Multiple Reading Levels: {title}\n\n" + "\n\n".join(levels)
        return SyntheticDocument(
            doc_id=doc_id,
            text=text,
            category="multi_level_writing",
            subcategory="graded_prose",
            difficulty="mixed",
            metadata={"topic": title, "type": "graded_explanations"}
        )

    def generate_grammar_corrections(self, doc_id: str) -> SyntheticDocument:
        """Common grammar errors paired with corrections and explanations."""
        examples = [
            [
                "Incorrect: Each of the students need to submit their assignment by Friday.",
                "Correct: Each of the students needs to submit his or her assignment by Friday. (Or: All students need to submit their assignments by Friday.)",
                "Explanation: 'Each' is a singular indefinite pronoun and requires a singular verb ('needs')."
            ],
            [
                "Incorrect: Running down the street, the hat blew off in the wind.",
                "Correct: As he was running down the street, his hat blew off in the wind.",
                "Explanation: The original sentence contained a dangling modifier. A hat cannot run down the street; the introductory participial phrase must clearly modify the subject."
            ],
            [
                "Incorrect: Between you and I, the decision has already been made.",
                "Correct: Between you and me, the decision has already been made.",
                "Explanation: 'Between' is a preposition and must be followed by objective pronouns ('me', 'him', 'her', 'us', 'them'), not subjective pronouns ('I')."
            ]
        ]

        ex = self.rng.choice(examples)
        text = "Grammar Correction & Analysis\n\n" + "\n".join(ex)
        return SyntheticDocument(
            doc_id=doc_id,
            text=text,
            category="grammar_and_syntax",
            subcategory="error_correction",
            difficulty="intermediate",
            metadata={"type": "grammar_repair"}
        )

    def generate_conversations(self, doc_id: str) -> SyntheticDocument:
        """Multi-turn realistic dialogue demonstrating formal and casual spoken English."""
        dialogues = [
            [
                "Dr. Aris: Good morning, Clara. Did the laboratory analysis yield any conclusive data regarding the soil samples?",
                "Clara: Good morning, Dr. Aris. Yes, the spectrometer results confirmed elevated trace mineral concentrations, particularly iron and magnesium.",
                "Dr. Aris: That aligns with our initial hypothesis. Shall we draft the summary report for the committee this afternoon?",
                "Clara: Absolutely. I will compile the chart visualizations beforehand so we can review them together."
            ],
            [
                "Maya: Have you decided which route we should take for the weekend hike?",
                "Julian: I was comparing the ridge trail with the river path. The ridge trail offers panoramic views, but it's quite steep.",
                "Maya: I don't mind a challenge as long as weather conditions stay clear. What does the afternoon forecast say?",
                "Julian: Mostly sunny with a light breeze. We should probably head out early to avoid the midday heat."
            ]
        ]

        lines = self.rng.choice(dialogues)
        text = "Spoken English Conversation\n\n" + "\n".join(lines)
        return SyntheticDocument(
            doc_id=doc_id,
            text=text,
            category="dialogue_and_discourse",
            subcategory="multi_turn_conversation",
            difficulty="intermediate",
            metadata={"num_turns": len(lines), "type": "dialogue"}
        )

    def generate_punctuation_and_rhetoric(self, doc_id: str) -> SyntheticDocument:
        """Examples exercising complex punctuation: em-dashes, semicolons, parentheticals, colons."""
        samples = [
            "Punctuation Focus: Semicolons and Colons\n\nThe expedition faced unexpected hardships; nevertheless, the team persevered with quiet determination. They carried three essential qualities: unwavering resilience, technical expertise, and mutual respect.",
            "Punctuation Focus: Em-Dashes and Parentheticals\n\nThe ancient manuscript—discovered accidentally in a subterranean vault—contained intricate astronomical charts. These records (dating back over seven centuries) revealed sophisticated observations of lunar eclipses.",
            "Rhetorical Structures: Parallelism and Contrast\n\nNot only did the innovation reduce energy consumption, but it also significantly lowered operational costs. True wisdom consists not in knowing all the answers, but in understanding which questions are worth asking."
        ]

        text = self.rng.choice(samples)
        return SyntheticDocument(
            doc_id=doc_id,
            text=text,
            category="punctuation_and_rhetoric",
            subcategory="advanced_syntax",
            difficulty="advanced",
            metadata={"type": "punctuation_exercise"}
        )
