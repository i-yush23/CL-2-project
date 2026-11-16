import os
import sys
import json

# Add external/CL-1-project to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CL1_PATH = os.path.join(PROJECT_ROOT, "external", "CL-1-project")
if CL1_PATH not in sys.path:
    sys.path.insert(0, CL1_PATH)

from hinglish.features.phonetic_matcher import get_phonetic_hash
from hinglish.features.emoji_analyzer import (
    load_emoji_lexicon,
    is_emoji_token,
    extract_emojis,
)
from hinglish.statistical_models.lid_model import LIDModel
from hinglish.metrics.cmi_calculator import CmiCalculator


def load_transliteration_variants() -> dict[str, str]:
    path = os.path.join(CL1_PATH, "hinglish", "data", "transliteration_variants.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_slang_map() -> dict[str, str]:
    path = os.path.join(CL1_PATH, "hinglish", "data", "slang_map.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_cl1_emoji_lexicon() -> dict[str, list[str]]:
    path = os.path.join(CL1_PATH, "hinglish", "data", "emoji_lexicon.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
