import re


NUMBER_PLUS_OR_MINUS = r"\b\d+(?:\.\d+)?\s*±\s*\d+(?:\.\d+)?\b"
NUMBER_WITH_UNIT_CONCENTRATION = r"\b\d+(?:\.\d+)?\s*(?:pg|ng|µg|mg|g|mol|μM|nM|mM|pM|M)\b"
NUMBER_PER_VOLUME            = r"\b\d+(?:\.\d+)?\s*(?:pg|ng|µg|mg)\s*/\s*(?:m?L|μL|L)\b"
RATE_PER_TIME                = r"\b\d+(?:\.\d+)?\s*(?:cells?|pg|ng|µg|IU|U)\s*/\s*(?:s|sec|min|h|hr|d|day|wk|week)s?\b"
SCIENTIFIC_NOTATION          = r"\b\d+(?:\.\d+)?\s*×\s*10\^?-?\d+\b"                       
PERCENTAGE                   = r"\b\d+(?:\.\d+)?\s*%\b"
SAMPLE_SIZE_N                = r"\bn\s*=\s*\d+\b"
P_VALUE                      = r"\bp\s*[<≤]\s*0\.\d+\b"
CONFIDENCE_INTERVAL          = r"\bCI\s*\(?\d+\s*[-–]\s*\d+\)?\b"
TIME_DURATION                = r"\b\d+(?:\.\d+)?\s*(?:s|sec|seconds|min|minutes|h|hr|hours|d|day|days|wk|wks|week|weeks)\b"

PATTERNS = [
    NUMBER_PLUS_OR_MINUS,
    NUMBER_WITH_UNIT_CONCENTRATION,
    NUMBER_PER_VOLUME,
    RATE_PER_TIME,
    SCIENTIFIC_NOTATION,
    PERCENTAGE,
    SAMPLE_SIZE_N,
    P_VALUE,
    CONFIDENCE_INTERVAL,
    TIME_DURATION,
]

COMPILLED_PATTERNS = [re.compile(p, re.I) for p in PATTERNS]