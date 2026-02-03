"""
Configuration et mappings pour le projet AIF Movie Recommender
"""

# Mapping des genres (12 classes)
GENRES_MAPPING = {
    "horror": 0,
    "thriller": 1,
    "romance": 2,
    "action": 3,
    "comedy": 4,
    "drama": 5,
    "sci-fi": 6,
    "fantasy": 7,
    "documentary": 8,
    "animation": 9,
    "crime": 10,
    "adventure": 11
}

# Mapping inverse (index -> genre)
GENRES_MAPPING_INV = {
    0: "horror",
    1: "thriller",
    2: "romance",
    3: "action",
    4: "comedy",
    5: "drama",
    6: "sci-fi",
    7: "fantasy",
    8: "documentary",
    9: "animation",
    10: "crime",
    11: "adventure"
}

# Liste des genres
GENRES_LIST = list(GENRES_MAPPING.keys())
NUM_CLASSES = len(GENRES_MAPPING)
