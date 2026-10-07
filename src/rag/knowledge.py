"""Knowledge base for the Tier-1 parser.

Maps the five target properties to the words people actually use, and holds a
small corpus of exemplar phrases (phrase -> property + direction) that the
retrieval layer matches novel wording against. No torch, no downloads.
"""

# Open Polymer Challenge property keys, with units and human aliases.
PROPERTY_INFO = {
    "Tg": {
        "name": "glass transition temperature",
        "unit": "C",
        "aliases": ["tg", "glass transition", "glass-transition", "softening",
                    "thermal stability", "heat resistance", "heat resistant",
                    "thermally stable", "service temperature", "melting"],
    },
    "FFV": {
        "name": "fractional free volume",
        "unit": "",
        "aliases": ["ffv", "free volume", "fractional free volume", "porosity",
                    "permeability", "permeable", "gas separation", "membrane"],
    },
    "Tc": {
        "name": "thermal conductivity",
        "unit": "",
        "aliases": ["tc", "thermal conductivity", "conduct heat",
                    "heat conduction", "conductive"],
    },
    "Density": {
        "name": "density",
        "unit": "g/cm3",
        "aliases": ["density", "dense", "lightweight", "light weight", "heavy",
                    "light", "mass per volume"],
    },
    "Rg": {
        "name": "radius of gyration",
        "unit": "A",
        "aliases": ["rg", "radius of gyration", "chain size", "coil size",
                    "chain dimension", "molecular size", "compact chain",
                    "extended chain"],
    },
}

# Words that set a direction. "high" -> steer the property up, "low" -> down.
HIGH_WORDS = ["high", "higher", "large", "larger", "big", "great", "greater",
              "strong", "strongest", "increased", "increase", "maximize",
              "maximal", "max", "elevated", "rich", "dense", "heavy",
              "stable", "resistant", "rigid", "stiff", "hard", "extended"]
LOW_WORDS = ["low", "lower", "small", "smaller", "little", "weak", "weakest",
             "decreased", "decrease", "minimize", "minimal", "min", "reduced",
             "poor", "light", "lightweight", "soft", "flexible", "compact"]
MID_WORDS = ["moderate", "medium", "mid", "average", "intermediate", "balanced"]

# Comparators for explicit numeric constraints, longest-match first.
COMPARATORS = [
    ("at least", ">="), ("no less than", ">="), ("greater than or equal", ">="),
    ("greater than", ">"), ("more than", ">"), ("larger than", ">"),
    ("above", ">"), ("over", ">"), ("exceeding", ">"), ("exceed", ">"),
    ("at most", "<="), ("no more than", "<="), ("less than or equal", "<="),
    ("less than", "<"), ("smaller than", "<"), ("below", "<"), ("under", "<"),
    ("up to", "<="),
    (">=", ">="), ("<=", "<="), (">", ">"), ("<", "<"),
    ("around", "~"), ("about", "~"), ("roughly", "~"), ("approximately", "~"),
    ("near", "~"), ("of", "~"), ("equal to", "~"), ("equals", "~"), ("=", "~"),
]

# Retrieval corpus: exemplar phrase -> (property, direction in {high,low,mid}).
# The retriever matches each clause to its nearest exemplar when no property
# keyword is found outright, so paraphrases still resolve.
EXEMPLARS = [
    ("high glass transition temperature", "Tg", "high"),
    ("heat resistant polymer", "Tg", "high"),
    ("thermally stable at high temperature", "Tg", "high"),
    ("withstands high temperatures", "Tg", "high"),
    ("rigid and does not soften when heated", "Tg", "high"),
    ("low glass transition temperature", "Tg", "low"),
    ("soft and flexible at room temperature", "Tg", "low"),
    ("rubbery elastomer", "Tg", "low"),
    ("high fractional free volume", "FFV", "high"),
    ("permeable gas separation membrane", "FFV", "high"),
    ("porous and open structure", "FFV", "high"),
    ("low free volume dense packing", "FFV", "low"),
    ("impermeable barrier film", "FFV", "low"),
    ("high thermal conductivity", "Tc", "high"),
    ("conducts heat well", "Tc", "high"),
    ("thermally insulating", "Tc", "low"),
    ("low thermal conductivity", "Tc", "low"),
    ("high density material", "Density", "high"),
    ("heavy dense polymer", "Density", "high"),
    ("low density lightweight material", "Density", "low"),
    ("light and low mass per volume", "Density", "low"),
    ("large radius of gyration", "Rg", "high"),
    ("extended open chain conformation", "Rg", "high"),
    ("small compact chain", "Rg", "low"),
    ("low radius of gyration", "Rg", "low"),
]
