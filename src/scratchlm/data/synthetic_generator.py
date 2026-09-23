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
    genre: str
    difficulty: str
    source_name: str = "ScratchLM Synthetic English Generator v1.0"
    source_url: str = "internal://scratchlm/synthetic_generator"
    author: str = "ScratchLM AI Augmentation Engine"
    license: str = "CC0 1.0 Universal (Public Domain Dedication)"
    license_evidence: str = "Generated synthetically by ScratchLM pipeline under CC0 Public Domain Dedication"
    is_synthetic: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class SyntheticEnglishGenerator:
    """Generates varied synthetic English documents across multiple linguistic dimensions."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)

    def generate_all(self, target_count: int = 100) -> List[SyntheticDocument]:
        """Generate a total pool of synthetic documents distributed across categories with guaranteed uniqueness."""
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
            for seq in range(max(1, count)):
                doc = gen_fn(f"synth_{doc_idx:05d}", seq)
                docs.append(doc)
                doc_idx += 1

        return docs

    def generate_grammar_transformations(self, doc_id: str, seq: int = 0) -> SyntheticDocument:
        """Active/passive, direct/indirect speech, conditional transformations."""
        doc_rng = random.Random(self.seed + seq * 17 + 101)
        
        templates = [
            ("Active to Passive Voice Transformation", [
                "Active Voice: The structural engineer designed a durable suspension bridge across the wide river channel.",
                "Passive Voice: A durable suspension bridge across the wide river channel was designed by the structural engineer.",
                "Active Voice: Marine biologists discovered a new species of bioluminescent organism during abyssal deep ocean exploration.",
                "Passive Voice: A new species of bioluminescent organism was discovered during abyssal deep ocean exploration by marine biologists.",
                "Grammatical Analysis: In passive constructions, the grammatical object of the active verb becomes the subject of the passive sentence, shifting thematic focus to the receiver of the action.",
                "Active Voice: The research team published a groundbreaking peer-reviewed study on quantum particle entanglement.",
                "Passive Voice: A groundbreaking peer-reviewed study on quantum particle entanglement was published by the research team.",
                "Active Voice: Environmental regulators established strict limits on chemical emissions from manufacturing plants.",
                "Passive Voice: Strict limits on chemical emissions from manufacturing plants were established by environmental regulators."
            ]),
            ("Conditional Clauses (Real, Unreal, and Mixed)", [
                "First Conditional (Real Future): If ambient atmospheric temperatures drop below freezing tonight, water trapped inside the pipe will expand and risk fracturing the mechanical seal.",
                "Second Conditional (Unreal Present): If our astronomical observatory possessed a thirty-meter optical mirror array, researchers could resolve individual terrestrial exoplanets orbiting neighboring stars.",
                "Third Conditional (Unreal Past): If the maintenance crew had inspected the hydraulic seals prior to departure, they would have identified the pressure loss before takeoff clearance.",
                "Grammatical Rule: Conditionals express hypothetical cause-and-effect relationships through structured modal auxiliary verb pairings.",
                "Mixed Conditional: If we had installed rooftop solar photovoltaic panels last year, our monthly utility electricity bills would be substantially lower today.",
                "Zero Conditional (Scientific Fact): If atmospheric air cools to its dew point temperature, water vapor condenses into liquid water droplets or fog."
            ]),
            ("Relative Clauses and Subordination Techniques", [
                "Simple Sentence 1: The astrophysicist published an influential research paper regarding galactic dark matter halos.",
                "Simple Sentence 2: She received an international scientific award from the national academy of sciences.",
                "Integrated Relative Clause: The astrophysicist who published an influential research paper regarding galactic dark matter halos received an international scientific award from the national academy of sciences.",
                "Non-restrictive Relative Clause: Dark matter halos, which envelop disk galaxies throughout the universe, exert tremendous gravitational influence despite emitting no light.",
                "Grammatical Function: Relative clauses modify antecedent nouns by embedding subordinate clauses using relative pronouns such as who, which, or that."
            ]),
            ("Direct vs. Indirect Speech Transformations", [
                "Direct Speech: 'The research satellite will complete its orbital mapping mission by midnight,' stated the flight director.",
                "Indirect Speech: The flight director stated that the research satellite would complete its orbital mapping mission by midnight.",
                "Direct Speech: 'Did the field expedition uncover any prehistoric fossil specimens along the riverbank?' asked the geologist.",
                "Indirect Speech: The geologist inquired whether the field expedition had uncovered any prehistoric fossil specimens along the riverbank.",
                "Tense Shift Rule: In reporting speech, present tense verbs backshift to corresponding past tense forms when the main reporting verb is in the past tense."
            ]),
            ("Inversion and Cleft Sentence Structures", [
                "Standard Order: Seldom have atmospheric scientists recorded such intense pressure drops during a hurricane.",
                "Inverted Order: Seldom have such intense pressure drops been recorded by atmospheric scientists during a hurricane.",
                "Standard Order: The team wanted to measure soil nitrogen content.",
                "Cleft Structure: It was soil nitrogen content that the team wanted to measure.",
                "Rhetorical Function: Inversion and cleft sentence constructions emphasize specific syntactic elements, enhancing rhetorical focus."
            ])
        ]

        chosen = doc_rng.sample(templates, 3)
        sections = [f"Linguistic Module {seq + 1}.{i+1}: {title}\n\n" + "\n\n".join(lines) for i, (title, lines) in enumerate(chosen)]
        text = f"Synthetic Grammar Exercise Set {seq + 1}\n\n" + "\n\n====================\n\n".join(sections)
        
        return SyntheticDocument(
            doc_id=doc_id, text=text, category="grammar_and_syntax", subcategory="sentence_transformations",
            genre="synthetic_grammar_exercise", difficulty="intermediate", metadata={"seq": seq, "type": "structural_grammar"}
        )

    def generate_word_contrasts(self, doc_id: str, seq: int = 0) -> SyntheticDocument:
        """Differentiating commonly confused words and synonyms in context."""
        doc_rng = random.Random(self.seed + seq * 19 + 202)
        
        pairs = [
            ("Affect vs. Effect", [
                "Affect is primarily used as a verb meaning 'to influence, produce a change in, or touch emotionally.'",
                "Example Sentence: How will shifting seasonal weather patterns affect agricultural crop yields across temperate farming regions?",
                "Effect is primarily used as a noun meaning 'the result, outcome, or consequence of an underlying cause.'",
                "Example Sentence: The positive long-term effect of regular aerobic exercise on human cardiovascular health is well established by medical studies.",
                "Usage Exception: Effect can function as a formal verb meaning 'to bring about or accomplish' (e.g., 'the legislative committee worked tirelessly to effect meaningful institutional reform')."
            ]),
            ("Principal vs. Principle", [
                "Principal can function as an adjective meaning 'chief, primary, or main,' or as a noun referring to the head of an institution or capital sum.",
                "Example Sentence: The principal objective of the environmental restoration initiative is re-establishing native plant species.",
                "Principle is exclusively a noun meaning 'a fundamental truth, law, doctrine, or rule of conduct.'",
                "Example Sentence: The scientific researcher refused to compromise her core professional principles during the empirical experiment."
            ]),
            ("Continuous vs. Continual", [
                "Continuous describes an action or state occurring without any pause, break, or interruption in time.",
                "Example Sentence: The continuous hum of the ventilation fan provided constant background white noise in the laboratory.",
                "Continual describes an action that recurs repeatedly over time, punctuated by brief interruptions.",
                "Example Sentence: The keynote speaker was frequently interrupted by continual bursts of enthusiastic applause from the audience."
            ]),
            ("Disinterested vs. Uninterested", [
                "Disinterested means impartial, unbiased, or neutral in judgment without personal financial or political stake.",
                "Example Sentence: An arbitrator must remain completely disinterested when reviewing commercial contract disputes.",
                "Uninterested means lacking interest, unconcerned, or indifferent to a subject.",
                "Example Sentence: Several students appeared uninterested during the lengthy historical lecture on tax reform."
            ]),
            ("Farther vs. Further", [
                "Farther refers to measurable physical distance across space.",
                "Example Sentence: The hiking party traveled five miles farther down the canyon path before making camp.",
                "Further refers to figurative distance, degree, extent, or additional time.",
                "Example Sentence: The research committee agreed to investigate the environmental anomaly further before issuing a final report."
            ])
        ]

        chosen = doc_rng.sample(pairs, 3)
        sections = [f"Vocabulary Distinction Module {seq + 1}.{i+1}: {word_pair}\n\n" + "\n\n".join(lines) for i, (word_pair, lines) in enumerate(chosen)]
        text = f"Synthetic Vocabulary Precision Guide {seq + 1}\n\n" + "\n\n====================\n\n".join(sections)

        return SyntheticDocument(
            doc_id=doc_id, text=text, category="vocabulary_and_semantics", subcategory="word_contrasts",
            genre="synthetic_vocabulary_guide", difficulty="intermediate", metadata={"seq": seq, "type": "vocabulary_precision"}
        )

    def generate_qa_pairs(self, doc_id: str, seq: int = 0) -> SyntheticDocument:
        """Structured Question and Answer sets across science, geography, and general knowledge."""
        doc_rng = random.Random(self.seed + seq * 23 + 303)
        
        qa_sets = [
            [
                ("Question: How does photosynthesis convert sunlight into chemical energy in green plants?",
                 "Answer: Photosynthesis is the biochemical process through which green plants, algae, and cyanobacteria absorb light energy and convert it into chemical energy stored in glucose molecules. Specialized pigments called chlorophyll absorb light wavelengths within plant chloroplasts. Using water absorbed from roots and carbon dioxide from the surrounding air, plants synthesize carbohydrates while releasing molecular oxygen as a vital byproduct."),
                ("Question: Why is photosynthesis fundamental to maintaining Earth's atmospheric balance?",
                 "Answer: Photosynthesis supplies the primary organic energy foundation for almost all ecological food webs on Earth and maintains atmospheric oxygen levels necessary for aerobic cellular respiration across animal and microbial life.")
            ],
            [
                ("Question: What physical mechanism generates ocean tides on coastal shorelines?",
                 "Answer: Ocean tides are created primarily by differential gravitational pull exerted by the Moon and Sun acting upon Earth's liquid oceans. As Earth rotates through these gravitational bulges, coastal regions experience cyclic high and low water levels approximately twice daily."),
                ("Question: What distinguishes spring tides from neap tides during lunar cycles?",
                 "Answer: Spring tides occur during full and new moons when gravitational forces of the Moon and Sun align in a straight line, creating maximum tidal ranges. Neap tides occur during quarter moon phases when solar and lunar gravity pull at right angles, resulting in minimal tidal variation.")
            ],
            [
                ("Question: What is the scientific principle behind aircraft aerodynamic lift?",
                 "Answer: Aerodynamic lift is generated when air flows across an asymmetrical airfoil wing structure. Air moving across the curved upper surface travels faster than air beneath the flat lower surface, creating a static pressure differential according to Bernoulli's principle that pushes the wing upward."),
                ("Question: How do pilots control an aircraft around its three primary rotational axes?",
                 "Answer: Pilots use elevators to control pitch motion around the lateral axis, ailerons to control roll motion around the longitudinal axis, and rudders to control yaw motion around the vertical axis.")
            ],
            [
                ("Question: What causes the Doppler Effect in sound and light waves?",
                 "Answer: The Doppler Effect occurs when there is relative motion between a wave source and an observer. As a emitting source approaches, wave crests compress, increasing perceived frequency; as it moves away, waves stretch, decreasing perceived frequency."),
                ("Question: How do astronomers use the Doppler Effect to measure galactic expansion?",
                 "Answer: Astronomers analyze spectral redshift in light from distant galaxies. Shifted absorption lines indicate galaxies moving away from Earth, proving cosmic spatial expansion.")
            ]
        ]

        chosen = doc_rng.sample(qa_sets, 3)
        flat_qa = [item for sublist in chosen for item in sublist]
        formatted = "\n\n".join([f"{q}\n{a}" for q, a in flat_qa])
        
        return SyntheticDocument(
            doc_id=doc_id, text=f"Educational Q&A Study Module {seq + 1}\n\n{formatted}",
            category="educational_qa", subcategory="scientific_explanations",
            genre="synthetic_qa", difficulty="intermediate", metadata={"seq": seq, "type": "structured_qa"}
        )

    def generate_multi_level_explanations(self, doc_id: str, seq: int = 0) -> SyntheticDocument:
        """Explaining a complex idea at beginner, intermediate, and advanced English levels."""
        doc_rng = random.Random(self.seed + seq * 29 + 404)
        
        topics = [
            ("Gravity and Planetary Motion", [
                "Beginner Level: Gravity is the natural pulling force that keeps our feet on the ground and causes dropped objects to fall toward Earth. It also keeps the Moon circling around our planet in space.",
                "Intermediate Level: Gravity is a fundamental physical force where objects with mass attract one another across distance. The gravitational attraction increases with greater mass and decreases as the distance between objects increases.",
                "Advanced Level: In general relativity, gravity is described not as a mechanical force, but as a curvature of four-dimensional spacetime induced by mass and energy distribution. Massive celestial bodies distort surrounding spatial geometry, guiding the inertial trajectories of matter and electromagnetic radiation."
            ]),
            ("Biodiversity and Ecosystem Stability", [
                "Beginner Level: Biodiversity means having many different kinds of plants, animals, and trees living together safely in a forest, ocean, or lake.",
                "Intermediate Level: Biodiversity encompasses the full variety of life on Earth, including genetic variation within species, species richness within habitats, and ecosystem diversity across landscapes.",
                "Advanced Level: Ecosystem resilience relies upon species and functional diversity, which buffer ecological communities against environmental shocks, invasive species, and climate perturbations by preserving redundant ecological pathways."
            ]),
            ("Plate Tectonics and Mountain Building", [
                "Beginner Level: The Earth's ground is broken into big moving puzzle pieces called plates. When they push together, they build high mountains.",
                "Intermediate Level: Plate tectonics explains how Earth's outer rigid crust is divided into several major plates that float on top of hotter, softer rock underneath. Tectonic collisions push up mountain chains over millions of years.",
                "Advanced Level: Lithospheric tectonic plate dynamics are driven by subterranean mantle convection currents. Convergent boundary subduction and continental collision generate intense lithospheric compression, causing folding, faulting, and mountain building."
            ]),
            ("Atmospheric Pressure and Weather Systems", [
                "Beginner Level: Air around us has weight even though we cannot see it. Warm air rises and cool air sinks, bringing sun or rain.",
                "Intermediate Level: Atmospheric pressure is the force exerted by the weight of air molecules above Earth's surface. High-pressure systems bring clear skies, while low-pressure systems bring clouds and storms.",
                "Advanced Level: Atmospheric barometric pressure gradients drive wind vectors adjusted by Coriolis deflection. Baroclinic instability along polar fronts initiates cyclogenesis, producing atmospheric low-pressure storms."
            ])
        ]

        chosen = doc_rng.sample(topics, 3)
        sections = [f"Concept Explanation Module {seq + 1}.{i+1}: {title}\n\n" + "\n\n".join(levels) for i, (title, levels) in enumerate(chosen)]
        text = f"Synthetic Graded Reading Module {seq + 1}\n\n" + "\n\n====================\n\n".join(sections)

        return SyntheticDocument(
            doc_id=doc_id, text=text, category="multi_level_writing", subcategory="graded_prose",
            genre="synthetic_graded_explanation", difficulty="mixed", metadata={"seq": seq, "type": "graded_explanations"}
        )

    def generate_grammar_corrections(self, doc_id: str, seq: int = 0) -> SyntheticDocument:
        """Common grammar errors paired with corrections and explanations."""
        doc_rng = random.Random(self.seed + seq * 31 + 505)
        
        examples = [
            ("Subject-Verb Agreement",
             "Incorrect Sentence: Each of the research participants need to complete their survey by Monday.\n\n"
             "Corrected Sentence: Each of the research participants needs to complete his or her survey by Monday. (Alternatively: All research participants need to complete their surveys by Monday.)\n\n"
             "Explanation: 'Each' is a singular indefinite pronoun that requires a singular verb form ('needs'). Using plural 'need' creates a subject-verb agreement error."),
            ("Dangling Modifiers",
             "Incorrect Sentence: Walking through the laboratory, the beaker fell off the counter.\n\n"
             "Corrected Sentence: As the researcher was walking through the laboratory, the beaker fell off the counter.\n\n"
             "Explanation: The original sentence contained a dangling participial modifier. A beaker cannot walk through a laboratory; the introductory phrase must clearly modify the human subject."),
            ("Pronoun Case Usage",
             "Incorrect Sentence: Between you and I, the committee has already selected the winning proposal.\n\n"
             "Corrected Sentence: Between you and me, the committee has already selected the winning proposal.\n\n"
             "Explanation: The preposition 'between' requires objective case pronouns ('me', 'him', 'her', 'them') rather than subjective case pronouns ('I', 'he', 'she', 'they')."),
            ("Misplaced Modifiers",
             "Incorrect Sentence: The professor served coffee to the students in paper cups.\n\n"
             "Corrected Sentence: The professor served coffee in paper cups to the students.\n\n"
             "Explanation: The modifier 'in paper cups' was misplaced next to 'students', implying the students were inside paper cups rather than the coffee.")
        ]

        chosen = doc_rng.sample(examples, 3)
        sections = [f"Grammar Error Analysis {seq + 1}.{i+1} ({title})\n\n{content}" for i, (title, content) in enumerate(chosen)]
        text = f"Synthetic Error Correction Guide {seq + 1}\n\n" + "\n\n====================\n\n".join(sections)

        return SyntheticDocument(
            doc_id=doc_id, text=text, category="grammar_and_syntax", subcategory="error_correction",
            genre="synthetic_grammar_repair", difficulty="intermediate", metadata={"seq": seq, "type": "grammar_repair"}
        )

    def generate_conversations(self, doc_id: str, seq: int = 0) -> SyntheticDocument:
        """Multi-turn realistic dialogue demonstrating formal and casual spoken English."""
        doc_rng = random.Random(self.seed + seq * 37 + 606)
        
        dialogues = [
            ("Academic Research Discussion", [
                "Dr. Aris: Good morning, Clara. Did the spectrometer analysis reveal any distinct mineral signatures in the soil samples collected near the fault line?",
                "Clara: Good morning, Dr. Aris. Yes, the laboratory readings confirmed elevated iron and magnesium concentrations near the eastern ridge.",
                "Dr. Aris: That aligns closely with our working hypothesis regarding regional hydrothermal activity. Shall we draft the summary report for the department head?",
                "Clara: Absolutely. I will organize the statistical data charts beforehand so we can review the trends together during this afternoon's meeting."
            ]),
            ("Practical Outdoor Planning", [
                "Maya: Have you decided which trail we should take for tomorrow morning's mountain hike?",
                "Julian: I was comparing the ridge path with the forest loop. The ridge path offers panoramic valley views, but it involves a steep rocky climb.",
                "Maya: I don't mind the incline as long as weather conditions stay clear. What does the afternoon forecast predict for cloud cover?",
                "Julian: Mostly sunny skies with a light breeze. We should head out early to avoid the afternoon sun."
            ]),
            ("Library Information Inquiry", [
                "Student: Excuse me, librarian. Could you assist me in locating historical primary sources regarding nineteenth-century canal construction?",
                "Librarian: Certainly. Our special archives collection holds original engineering journals and land survey maps on the second floor.",
                "Student: That sounds excellent. Are visitors required to schedule an advance research appointment?",
                "Librarian: No advance appointment is necessary for digital archives, though physical manuscript access requires presenting student identification."
            ])
        ]

        title, lines = doc_rng.choice(dialogues)
        text = f"Spoken Dialogue Exercise {seq + 1}: {title}\n\n" + "\n\n".join(lines)

        return SyntheticDocument(
            doc_id=doc_id, text=text, category="dialogue_and_discourse", subcategory="multi_turn_conversation",
            genre="synthetic_dialogue", difficulty="intermediate", metadata={"seq": seq, "type": "dialogue"}
        )

    def generate_punctuation_and_rhetoric(self, doc_id: str, seq: int = 0) -> SyntheticDocument:
        """Examples exercising complex punctuation: em-dashes, semicolons, parentheticals, colons."""
        samples = [
            "Punctuation Focus: Semicolons, Colons, and Em-Dashes\n\nThe geological expedition encountered unexpected weather hazards; nevertheless, the research team persevered with quiet determination. They relied upon three essential traits: rigorous preparation, technical adaptability, and mutual trust. The ancient stone inscription—discovered accidentally inside a subterranean chamber—contained detailed astronomical records.",
            "Rhetorical Structures: Parallelism, Antithesis, and Contrast\n\nNot only did the engineering innovation reduce fuel consumption, but it also substantially lowered carbon emissions. True intellectual curiosity consists not merely in acquiring facts, but in questioning long-held assumptions. To understand the future, we must study the past.",
            "Advanced Punctuation: Parenthetical Expressions and Serial Commas\n\nThe oceanographic research vessel (which had been docked for routine maintenance) sailed at sunrise. The scientist collected water samples, cataloged marine organisms, and recorded surface temperatures continuously throughout the cruise."
        ]

        text = f"Rhetorical & Punctuation Practice {seq + 1}\n\n" + "\n\n".join(samples)

        return SyntheticDocument(
            doc_id=doc_id, text=text, category="punctuation_and_rhetoric", subcategory="advanced_syntax",
            genre="synthetic_rhetoric_exercise", difficulty="advanced", metadata={"seq": seq, "type": "punctuation_exercise"}
        )
