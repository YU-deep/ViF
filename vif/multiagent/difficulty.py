import re
from dataclasses import dataclass

@dataclass
class DifficultyScore:
    score: float
    buckets: str

QUESTION_PATTERNS = {
    'why': re.compile(r'\\bwhy\\b', re.I),
    'how': re.compile(r'\\bhow\\b', re.I),
    'count': re.compile(r'\\bhow many|count\\b', re.I),
    'math': re.compile(r'\\badd|sum|minus|difference|average\\b', re.I),
    'ocr': re.compile(r'\\bread|text|word|character|number\\b', re.I),
}

def estimate_difficulty(instruction: str) -> DifficultyScore:
    s = instruction.strip()
    base = 0.2 if len(s) < 40 else 0.4 if len(s) < 120 else 0.6
    bonus = 0.0
    for rx in QUESTION_PATTERNS.values():
        if rx.search(s): bonus += 0.15
    score = max(0.0, min(1.0, base + bonus))
    bucket = 'easy' if score < 0.33 else ('medium' if score < 0.66 else 'hard')
    return DifficultyScore(score=score, buckets=bucket)
