"""Render 빌드 시 원본 CSV 압축을 풀고 서버용 한글 폰트를 준비합니다."""
import gzip
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
csv_file = ROOT / "data" / "age.csv"
compressed = ROOT / "data" / "age.csv.gz"
if not csv_file.exists():
    csv_file.write_bytes(gzip.decompress(compressed.read_bytes()))

font = ROOT / "fonts" / "NanumGothic-Regular.ttf"
font.parent.mkdir(exist_ok=True)
if not font.exists():
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/google/fonts/main/ofl/nanumgothic/NanumGothic-Regular.ttf", font)
print("CSV와 한글 폰트 준비 완료")
