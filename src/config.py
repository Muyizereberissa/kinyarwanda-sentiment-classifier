from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

AFRISENTI_KIN_BASE_URL = (
    "https://raw.githubusercontent.com/afrisenti-semeval/afrisent-semeval-2023/main/data/kin"
)

LANGUAGE = "Kinyarwanda"
ISO_CODE = "kin"
LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

# Default transformer checkpoint (African-language BERT; fine-tuned on AfriSenti kin)
# Pretrained on Gahuza (Kinyarwanda/Kirundi mix) among other African languages (Ogueji et al., 2021)
DEFAULT_TRANSFORMER = "castorini/afriberta_base"

RANDOM_SEED = 42
