"""경로, 연령층, 시연용 지표 기준과 API 환경변수."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=False)
DATA_PATH = BASE_DIR / "data" / "age.csv"
MAX_REGIONS = 5
AGE_GROUPS = {
    "유소년층": (0, 14), "청년층": (15, 29), "중년층": (30, 49),
    "장년층": (50, 64), "노년층": (65, None),
}
# 과제 시연용 임의 기준입니다. 공식 등급 또는 전국 평균을 뜻하지 않습니다.
AGING_THRESHOLDS = (14, 20)
YOUTH_THRESHOLDS = (15, 25)


def api_settings(provider):
    if provider == "ChatGPT":
        return os.getenv("OPENAI_API_KEY", "").strip(), os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    if provider == "Gemini":
        return os.getenv("GEMINI_API_KEY", "").strip(), os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    return "", "local"


def level(value, thresholds):
    low, high = thresholds
    return "낮음" if value < low else "보통" if value < high else "높음"
