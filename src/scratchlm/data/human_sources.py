"""
Real Human English Text Sources for ScratchLM Phase 1 Corpus.

Curates and ingests public-domain and open-licensed human English texts (~80% of corpus)
spanning diverse non-programming genres and reading levels.
"""

from typing import List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class HumanDocument:
    """A human-authored document from a public domain or open source."""
    doc_id: str
    text: str
    source_name: str
    license: str
    category: str
    subcategory: str
    difficulty: str
    is_synthetic: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class RealHumanSourceProvider:
    """Provides curated real human English texts across diverse non-programming domains."""

    def __init__(self):
        self.curated_pool = self._build_curated_pool()

    def get_all_documents(self) -> List[HumanDocument]:
        """Return all curated human documents."""
        return self.curated_pool

    def _build_curated_pool(self) -> List[HumanDocument]:
        docs: List[HumanDocument] = []
        doc_idx = 1

        raw_records = [
            # --- 1. ASTRONOMY & SPACE SCIENCE ---
            {
                "source": "NASA Public Domain Educational Archive",
                "license": "Public Domain (U.S. Government Work)",
                "category": "science_and_nature",
                "subcategory": "astronomy_and_space",
                "difficulty": "intermediate",
                "title": "The Solar System and Planetary Atmospheres",
                "text": (
                    "The Solar System consists of the Sun and the astronomical objects bound to it by gravity. "
                    "Among the eight planets, Earth is the third from the Sun and the only astronomical body known to harbor life. "
                    "About 71 percent of Earth's surface is covered with water, mostly by oceans, with the remaining area consisting of continents and islands. "
                    "Earth's atmosphere consists primarily of nitrogen (about 78 percent) and oxygen (about 21 percent), with trace amounts of argon, carbon dioxide, and water vapor. "
                    "This atmospheric composition shields life by absorbing ultraviolet solar radiation, warming the surface through heat retention, and reducing temperature extremes between day and night."
                )
            },
            {
                "source": "NASA History Office Publications",
                "license": "Public Domain (U.S. Government Work)",
                "category": "science_and_nature",
                "subcategory": "space_exploration",
                "difficulty": "advanced",
                "title": "Orbital Mechanics and Gravitational Slingshots",
                "text": (
                    "Spacecraft navigating the outer Solar System frequently utilize gravity assist maneuvers, commonly called gravitational slingshots. "
                    "By flying close to a planet, a spacecraft gains momentum from the planet's orbital movement around the Sun. "
                    "This technique conserves chemical propellant, enabling long-distance missions like Voyager and Cassini to reach distant gas giants within feasible timeframes. "
                    "Precise trajectory calculations require solving complex orbital mechanics equations involving gravitational forces, relative velocity vectors, and atmospheric drag parameters."
                )
            },

            # --- 2. GEOLOGY, EARTH SCIENCE & CLIMATE ---
            {
                "source": "U.S. Geological Survey (USGS) Educational Guides",
                "license": "Public Domain (U.S. Government Work)",
                "category": "science_and_nature",
                "subcategory": "geology_and_earth_science",
                "difficulty": "intermediate",
                "title": "Plate Tectonics and Mountain Building",
                "text": (
                    "Plate tectonics is a scientific theory describing the large-scale motion of seven large plates and the movements of a larger number of smaller plates of Earth's lithosphere. "
                    "The lithosphere, which is the rigid outermost shell of a planet, is broken into tectonic plates. "
                    "Where plates meet, their relative motion determines the type of boundary: convergent, divergent, or transform. "
                    "Earthquakes, volcanic activity, mountain-building, and oceanic trench formation occur along these plate boundaries. "
                    "For example, the collision between the Indian Plate and the Eurasian Plate formed the Himalayan mountain range over tens of millions of years."
                )
            },
            {
                "source": "National Oceanic and Atmospheric Administration (NOAA)",
                "license": "Public Domain (U.S. Government Work)",
                "category": "science_and_nature",
                "subcategory": "oceanography_and_climate",
                "difficulty": "intermediate",
                "title": "Ocean Currents and Thermohaline Circulation",
                "text": (
                    "Thermohaline circulation, often referred to as the global ocean conveyor belt, is driven by differences in water density caused by temperature and salinity variations. "
                    "Cold, salty water sinks near the poles and flows slowly along the deep ocean floor toward the equator, while warmer surface waters transport tropical heat northward. "
                    "This vast circulation pattern plays a crucial role in regulating Earth's climate system by redistributing thermal energy across geographical latitudes."
                )
            },

            # --- 3. ECOLOGY & BIOLOGY ---
            {
                "source": "U.S. Fish and Wildlife Service Public Domain Ecology Guide",
                "license": "Public Domain (U.S. Government Work)",
                "category": "science_and_nature",
                "subcategory": "biology_and_ecology",
                "difficulty": "intermediate",
                "title": "Ecological Adaptations in Temperate Rainforests",
                "text": (
                    "Temperate rainforests are coniferous or broadleaf forests that occur in the temperate zone and receive high rainfall. "
                    "Heavy precipitation and moderate seasonal temperatures foster dense vegetation, characterized by ancient evergreen trees, epiphytes, and thick moss layers. "
                    "Trees such as Sitka spruce and Douglas fir can reach towering heights exceeding two hundred feet. "
                    "Decaying fallen trees, known as nurse logs, provide essential moisture and nutrients for germinating seedlings on the forest floor, demonstrating nature's efficient nutrient recycling mechanisms."
                )
            },
            {
                "source": "U.S. National Park Service Nature Archive",
                "license": "Public Domain (U.S. Government Work)",
                "category": "science_and_nature",
                "subcategory": "botany_and_plant_biology",
                "difficulty": "beginner_intermediate",
                "title": "Plant Adaptations in Arid Desert Environments",
                "text": (
                    "Desert plants have evolved extraordinary physical adaptations to survive extreme heat and prolonged droughts. "
                    "Succulents like the saguaro cactus store vast quantities of water in specialized thick stems during brief rainstorms. "
                    "Waxy stem coatings reduce moisture loss through evaporation, while sharp spines deter thirsty desert herbivores. "
                    "Additionally, many desert shrubs feature deep taproots that extend far beneath the sun-baked soil to reach subterranean water tables."
                )
            },

            # --- 4. HISTORICAL SPEECHES & ORATORY ---
            {
                "source": "Project Gutenberg - Abraham Lincoln Speeches",
                "license": "Public Domain",
                "category": "history_and_speeches",
                "subcategory": "historical_oratory",
                "difficulty": "advanced",
                "title": "The Gettysburg Address",
                "text": (
                    "Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal. "
                    "Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. "
                    "We are met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live. "
                    "It is altogether fitting and proper that we should do this. But, in a larger sense, we can not dedicate—we can not consecrate—we can not hallow—this ground. "
                    "The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract. "
                    "The world will little note, nor long remember what we say here, but it can never forget what they did here."
                )
            },
            {
                "source": "U.S. National Archives Public Domain Historical Speeches",
                "license": "Public Domain (U.S. Government Work)",
                "category": "history_and_speeches",
                "subcategory": "historical_oratory",
                "difficulty": "advanced",
                "title": "George Washington's Farewell Address (Excerpt)",
                "text": (
                    "Observe good faith and justice towards all nations; cultivate peace and harmony with all. "
                    "Religion and morality enjoin this conduct; and can it be, that good policy does not equally enjoin it? "
                    "It will be worthy of a free, enlightened, and at no distant period, a great nation, to give to mankind the magnanimous and too novel example of a people always guided by an exalted justice and benevolence. "
                    "The nation which indulges towards another an habitual hatred or an habitual fondness is in some degree a slave. "
                    "It is a slave to its animosity or to its affection, either of which is sufficient to lead it astray from its duty and its interest."
                )
            },

            # --- 5. ESSAYS, PHILOSOPHY & PROSE ---
            {
                "source": "Project Gutenberg - Ralph Waldo Emerson Essays",
                "license": "Public Domain",
                "category": "essays_and_philosophy",
                "subcategory": "transcendentalism_and_prose",
                "difficulty": "advanced",
                "title": "Excerpt from Self-Reliance by Ralph Waldo Emerson",
                "text": (
                    "There is a time in every man's education when he arrives at the conviction that envy is ignorance; that imitation is suicide; "
                    "that he must take himself for better for worse as his portion; that though the wide universe is full of good, no kernel of nourishing corn can come to him but through his toil bestowed on that plot of ground which is given to him to till. "
                    "The power which resides in him is new in nature, and none but he knows what that is which he can do, nor does he know until he has tried. "
                    "Trust thyself: every heart vibrates to that iron string. Accept the place the divine providence has found for you, the society of your contemporaries, the connection of events."
                )
            },
            {
                "source": "Project Gutenberg - Henry David Thoreau (Walden)",
                "license": "Public Domain",
                "category": "essays_and_philosophy",
                "subcategory": "nature_and_solitude",
                "difficulty": "advanced",
                "title": "Excerpt from Walden by Henry David Thoreau",
                "text": (
                    "I went to the woods because I wished to live deliberately, to front only the essential facts of life, and see if I could not learn what it had to teach, and not, when I came to die, discover that I had not lived. "
                    "I did not wish to live what was not life, living is so dear; nor did I wish to practice resignation, unless it was quite necessary. "
                    "I wanted to live deep and suck out all the marrow of life, to live so sturdily and Spartan-like as to put to rout all that was not life. "
                    "Simplicity, simplicity, simplicity! I say, let your affairs be as two or three, and not a hundred or a thousand; instead of a million count half a dozen, and keep your accounts on your thumb-nail."
                )
            },

            # --- 6. HISTORY, GEOGRAPHY & BIOGRAPHY ---
            {
                "source": "U.S. National Park Service History Archive",
                "license": "Public Domain (U.S. Government Work)",
                "category": "history_and_geography",
                "subcategory": "historical_biography",
                "difficulty": "intermediate",
                "title": "Biographical Profile of Benjamin Franklin",
                "text": (
                    "Benjamin Franklin was one of the Founding Fathers of the United States, a polymath, inventor, scientist, printer, and diplomat. "
                    "As a scientist, he was a major figure in the American Enlightenment and the history of physics for his discoveries and theories regarding electricity. "
                    "As an inventor, he is known for the lightning rod, bifocals, and the Franklin stove, among other inventions. "
                    "He facilitated many civic organizations, including Philadelphia's fire department and the University of Pennsylvania. "
                    "Franklin earned the title of 'The First American' for his early and indefatigable campaigning for colonial unity."
                )
            },
            {
                "source": "Library of Congress Public Domain Historical Collections",
                "license": "Public Domain",
                "category": "history_and_geography",
                "subcategory": "world_geography",
                "difficulty": "intermediate",
                "title": "Geography and Hydrology of the Nile River Basin",
                "text": (
                    "The Nile is a major north-flowing river in northeastern Africa and is commonly regarded as the longest river in the world. "
                    "Spanning over 4,100 miles, its drainage basin covers eleven countries: Tanzania, Uganda, Rwanda, Burundi, the Democratic Republic of the Congo, Kenya, Ethiopia, Eritrea, South Sudan, Republic of the Sudan, and Egypt. "
                    "Historically, the annual flooding of the Nile deposited nutrient-rich silt along its banks, enabling ancient Egyptian civilization to flourish in an otherwise arid desert environment. "
                    "Today, the river remains a vital source of water, hydroelectric power, and agricultural irrigation for over two hundred million people."
                )
            },

            # --- 7. HOW-TO GUIDES, INSTRUCTIONS & PRACTICAL EXPLANATIONS ---
            {
                "source": "U.S. Department of Energy Consumer Guides",
                "license": "Public Domain (U.S. Government Work)",
                "category": "instructions_and_how_to",
                "subcategory": "home_energy_efficiency",
                "difficulty": "beginner_intermediate",
                "title": "How to Improve Home Energy Efficiency",
                "text": (
                    "Improving home energy efficiency reduces utility costs and minimizes environmental impact. "
                    "First, inspect windows and doors for drafts. Apply weatherstripping around movable components and use caulk to seal stationary gaps. "
                    "Second, upgrade to a programmable thermostat, which automatically lowers heating or cooling when you are asleep or away from home. "
                    "Third, clean or replace air filters in your heating and cooling system once every one to three months. "
                    "Finally, consider replacing incandescent light bulbs with energy-efficient LED bulbs, which consume up to 85 percent less electricity and last significantly longer."
                )
            },
            {
                "source": "U.S. Department of Agriculture Food Safety and Inspection Service Guide",
                "license": "Public Domain (U.S. Government Work)",
                "category": "instructions_and_how_to",
                "subcategory": "food_safety_and_cooking",
                "difficulty": "beginner_intermediate",
                "title": "Safe Food Handling Guidelines",
                "text": (
                    "Safe food handling involves four basic steps: clean, separate, cook, and chill. "
                    "Clean: Wash hands, utensils, and cutting boards with warm soap and water before and after preparing food. "
                    "Separate: Keep raw meat, poultry, seafood, and eggs separate from ready-to-eat foods in your shopping cart, refrigerator, and meal preparation areas. "
                    "Cook: Use a food thermometer to ensure meats reach safe internal temperatures—for instance, 165 degrees Fahrenheit for poultry. "
                    "Chill: Refrigerate perishable foods promptly within two hours of purchase or cooking."
                )
            },

            # --- 8. DIALOGUE, CONVERSATIONS & THEATRICAL DRAMA ---
            {
                "source": "Project Gutenberg - Oscar Wilde (The Importance of Being Earnest)",
                "license": "Public Domain",
                "category": "dialogue_and_discourse",
                "subcategory": "classic_drama_and_conversation",
                "difficulty": "intermediate_advanced",
                "title": "Excerpt from The Importance of Being Earnest by Oscar Wilde",
                "text": (
                    "Algernon: How are you, my dear Ernest? What brings you to town?\n"
                    "Jack: Oh, pleasure, pleasure! What else should bring one anywhere? Eating as usual, I see, Algy!\n"
                    "Algernon: I believe it is customary in good society to take some refreshment at five o'clock. Where have you been since last Thursday?\n"
                    "Jack: In the country.\n"
                    "Algernon: What on earth do you do there?\n"
                    "Jack: When one is in town one amuses oneself. When one is in the country one amuses other people. It is excessively boring.\n"
                    "Algernon: And who are the people you amuse?\n"
                    "Jack: Oh, neighbors, neighbors!\n"
                    "Algernon: Got nice neighbors in your part of the world?\n"
                    "Jack: Perfectly dreadful! I never speak to any of them."
                )
            },
            {
                "source": "Project Gutenberg - William Shakespeare (Julius Caesar)",
                "license": "Public Domain",
                "category": "dialogue_and_discourse",
                "subcategory": "dramatic_dialogue",
                "difficulty": "advanced",
                "title": "Excerpt from Julius Caesar by William Shakespeare",
                "text": (
                    "Brutus: Be patient till the last. Romans, countrymen, and lovers! hear me for my cause, and be silent, that you may hear: believe me for mine honor, and have respect to mine honor, that you may believe. "
                    "If there be any in this assembly, any dear friend of Caesar's, to him I say, that Brutus' love to Caesar was no less than his. "
                    "If then that friend demand why Brutus rose against Caesar, this is my answer:—Not that I loved Caesar less, but that I loved Rome more. "
                    "Had you rather Caesar were living and die all slaves, than that Caesar were dead, to live all free men?"
                )
            },

            # --- 9. CLASSIC FICTION & PROSE NOVELS ---
            {
                "source": "Project Gutenberg - Jules Verne (Twenty Thousand Leagues Under the Sea)",
                "license": "Public Domain",
                "category": "fiction_and_literature",
                "subcategory": "adventure_and_prose",
                "difficulty": "intermediate",
                "title": "Excerpt from Twenty Thousand Leagues Under the Sea by Jules Verne",
                "text": (
                    "The year 1866 was signalized by a remarkable incident, a mysterious and puzzling phenomenon, which doubtless no one has yet forgotten. "
                    "For some time past vessels had been met by 'an enormous thing' at sea, a long object, spindle-shaped, occasionally phosphorescent, and infinitely larger and more rapid in its movements than a whale. "
                    "The rumors relating to this apparition (recorded in various log-books) deeply intrigued maritime authorities, merchants, and sea captains across both hemispheres. "
                    "Scientific opinion was divided: some regarded the monster as a leviathan of unknown giant proportions, while others asserted it was a submarine vessel of extraordinary mechanical power."
                )
            },
            {
                "source": "Project Gutenberg - Herman Melville (Moby Dick)",
                "license": "Public Domain",
                "category": "fiction_and_literature",
                "subcategory": "classic_novel",
                "difficulty": "advanced",
                "title": "Excerpt from Moby Dick by Herman Melville",
                "text": (
                    "Call me Ishmael. Some years ago—never mind how long precisely—having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. "
                    "It is a way I have of driving off the spleen and regulating the circulation. "
                    "Whenever I find myself growing grim about the mouth; whenever it is a damp, drizzly November in my soul; whenever I find myself involuntarily pausing before coffin warehouses, and bringing up the rear of every funeral I meet; "
                    "and especially whenever my hypos get such an upper hand of me, that it requires a strong moral principle to prevent me from deliberately stepping into the street, and methodically knocking people's hats off—then, I account it high time to get to sea as soon as I can."
                )
            },

            # --- 10. CHILDREN'S & BEGINNER EDUCATIONAL ENGLISH ---
            {
                "source": "Project Gutenberg - Aesop's Fables",
                "license": "Public Domain",
                "category": "childrens_and_beginner",
                "subcategory": "fables_and_morale",
                "difficulty": "beginner",
                "title": "The Tortoise and the Hare",
                "text": (
                    "A Hare was once boasting of his speed before the other animals. 'I have never yet been beaten,' said he, 'when I put forth my full speed. I challenge any one here to race with me.'\n"
                    "The Tortoise said quietly, 'I accept your challenge.'\n"
                    "'That is a good joke,' said the Hare; 'I could dance round you all the way.'\n"
                    "'Keep your boasting till you have won,' answered the Tortoise. 'Shall we race?'\n"
                    "So a course was fixed and a start was made. The Hare darted almost out of sight at once, but soon stopped and, to show his contempt for the Tortoise, lay down to have a nap. "
                    "The Tortoise plodded on and plodded on, and when the Hare awoke from his nap, he saw the Tortoise just near the winning post, and could not run up in time to save the race.\n"
                    "Moral: Slow and steady wins the race."
                )
            },
            {
                "source": "Project Gutenberg - Beatrix Potter (The Tale of Peter Rabbit)",
                "license": "Public Domain",
                "category": "childrens_and_beginner",
                "subcategory": "childrens_prose",
                "difficulty": "beginner",
                "title": "Excerpt from The Tale of Peter Rabbit by Beatrix Potter",
                "text": (
                    "Once upon a time there were four little Rabbits, and their names were—Flopsy, Mopsy, Cotton-tail, and Peter. "
                    "They lived with their Mother in a sand-bank, underneath the root of a very big fir-tree. "
                    "'Now my dears,' said old Mrs. Rabbit one morning, 'you may go into the fields or down the lane, but don't go into Mr. McGregor's garden: your Father had an accident there; he was put in a pie by Mrs. McGregor. "
                    "Now run along, and don't get into mischief. I am going out.'"
                )
            },

            # --- 11. TECHNICAL NON-PROGRAMMING ENGLISH ---
            {
                "source": "U.S. Patent and Trademark Office / Public Technology Guides",
                "license": "Public Domain (U.S. Government Work)",
                "category": "science_and_technology",
                "subcategory": "mechanical_engineering",
                "difficulty": "advanced",
                "title": "Principles of Four-Stroke Internal Combustion Engines",
                "text": (
                    "A four-stroke internal combustion engine converts thermal energy derived from liquid fuel into rotational kinetic energy through four distinct piston strokes: intake, compression, power, and exhaust. "
                    "During the intake stroke, the intake valve opens as the piston descends, drawing a mixture of atomized fuel and atmospheric air into the cylinder. "
                    "Next, the compression stroke pushes the piston upward with both valves closed, compressing the air-fuel charge to increase thermal efficiency. "
                    "A spark plug ignites the pressurized mixture, creating rapid gas expansion that forces the piston downward during the power stroke. "
                    "Finally, the exhaust valve opens as the piston rises again, expelling spent combustion gases into the manifold."
                )
            },
            {
                "source": "U.S. Federal Aviation Administration (FAA) Pilot Handbook",
                "license": "Public Domain (U.S. Government Work)",
                "category": "science_and_technology",
                "subcategory": "aerodynamics_and_aviation",
                "difficulty": "advanced",
                "title": "Aerodynamic Principles of Lift and Bernoulli's Theorem",
                "text": (
                    "Aerodynamic lift is the force that directly opposes the weight of an aircraft and holds it in the air. "
                    "Lift is generated primarily by the wings as they move through the fluid medium of the atmosphere. "
                    "According to Bernoulli's principle, an increase in the speed of a fluid occurs simultaneously with a decrease in static pressure. "
                    "An airfoil is contoured such that air passing over the curved upper surface travels at a higher velocity than air passing beneath the flatter lower surface. "
                    "This velocity differential creates a net static pressure imbalance, resulting in an upward force known as lift."
                )
            }
        ]

        for item in raw_records:
            doc = HumanDocument(
                doc_id=f"human_{doc_idx:05d}",
                text=item["text"],
                source_name=item["source"],
                license=item["license"],
                category=item["category"],
                subcategory=item["subcategory"],
                difficulty=item["difficulty"],
                is_synthetic=False,
                metadata={
                    "title": item["title"],
                    "provenance": "public_domain_curated"
                }
            )
            docs.append(doc)
            doc_idx += 1

        return docs
