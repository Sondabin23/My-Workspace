"""CSV 구조 자동 탐색과 인구 통계. 시/구/동 원본 행을 각각 분석합니다."""
import hashlib
import csv
import io
import re

import numpy as np
import pandas as pd

from config import AGE_GROUPS, AGING_THRESHOLDS, YOUTH_THRESHOLDS, level


def read_csv_bytes(raw):
    if not raw or not raw.strip():
        raise ValueError("CSV 파일이 비어 있습니다.")
    # 문자열로 읽어 천 단위 쉼표, 빈 셀과 행정구역 코드를 보존합니다.
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        try:
            header = next(csv.reader(io.StringIO(text)))
            header = [c.strip() for c in header]
            if len(header) != len(set(header)):
                raise ValueError("중복 열 이름이 있습니다. 열 이름을 구분해 주세요.")
            frame = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise ValueError("CSV 형식이 잘못되었습니다. 구분자와 헤더를 확인하세요.") from exc
        if frame.empty:
            raise ValueError("CSV에 데이터 행이 없습니다.")
        frame.columns = [str(c).strip() for c in frame.columns]
        if frame.columns.duplicated().any():
            raise ValueError("중복 열 이름이 있습니다. 열 이름을 구분해 주세요.")
        return frame, encoding
    raise ValueError("UTF-8, UTF-8-SIG, CP949, EUC-KR로 읽을 수 없습니다.")


def discover_schema(frame):
    # 다른 이름을 쓰는 CSV라면 이 후보와 아래 연령 정규식만 수정하면 됩니다.
    region_columns = [c for c in frame.columns if any(
        token in c for token in ("행정구역", "지역명", "행정동", "지역", "읍면동명"))]
    if len(region_columns) != 1:
        raise ValueError("지역명 열을 하나로 확인할 수 없습니다. discover_schema의 후보를 수정하세요.")
    series = {}
    for column in frame.columns:
        match = re.search(r"(?<!\d)(\d{1,3})\s*세\s*(이상)?(?:\s*인구(?:수)?)?\s*$", column)
        if not match:
            continue
        age = int(match.group(1))
        if age > 130:
            raise ValueError("130세를 넘는 연령 열이 있습니다. 데이터 정의를 확인하세요.")
        prefix = column[:match.start()].rstrip("_ -") or "연령별 인구"
        entry = series.setdefault(prefix, {"columns": {}, "open_age": None})
        if age in entry["columns"]:
            raise ValueError(f"{prefix}: {age}세 열이 중복됩니다.")
        entry["columns"][age] = column
        if match.group(2):
            if entry["open_age"] is not None:
                raise ValueError("'세 이상' 열은 마지막 연령에 하나만 있어야 합니다.")
            entry["open_age"] = age
    if not series:
        raise ValueError("0세, 1세, …, 100세 이상 형태의 연령 열을 찾지 못했습니다.")
    for prefix, entry in series.items():
        ages = sorted(entry["columns"])
        if ages != list(range(ages[-1] + 1)) or ages[-1] < 65:
            raise ValueError(f"{prefix}: 0세부터 65세 이상까지 연속된 단일 연령 열이 필요합니다.")
        if entry["open_age"] is not None and entry["open_age"] != ages[-1]:
            raise ValueError("'세 이상'은 마지막 연령 열이어야 합니다.")
        if entry["open_age"] is not None and entry["open_age"] < 65:
            raise ValueError("65세 미만에서 시작하는 개방 연령 구간은 지원하지 않습니다.")
        total_candidates = [c for c in frame.columns if "총인구" in c
                            and (prefix == "연령별 인구" or c.startswith(prefix + "_"))]
        entry["total_column"] = total_candidates[0] if len(total_candidates) == 1 else None
    return region_columns[0], series


def load_data(raw, selected_series=None):
    frame, encoding = read_csv_bytes(raw)
    region_column, series = discover_schema(frame)
    # 같은 파일에 남/여/총계 또는 여러 월이 있으면 화면에서 계열을 선택합니다.
    default = next((s for s in series if re.search(r"(?:^|_)총(?:_|$)", s)), next(iter(series)))
    selected_series = selected_series or default
    if selected_series not in series:
        raise ValueError("선택한 연령 데이터 계열이 없습니다.")
    schema = series[selected_series]
    names = frame[region_column].str.strip()
    if names.eq("").any() or names.duplicated().any():
        raise ValueError("빈 지역명 또는 동일한 지역 행이 있습니다. 원본을 확인하세요.")
    ages = sorted(schema["columns"])
    columns = [schema["columns"][age] for age in ages]
    numeric = frame[columns].apply(lambda c: pd.to_numeric(
        c.str.strip().str.replace(",", "", regex=False), errors="coerce"))
    numeric.columns = ages
    valid = np.isfinite(numeric) & numeric.ge(0) & numeric.eq(np.floor(numeric))
    bad_rows = (~valid).any(axis=1)
    total_column = schema["total_column"]
    totals = None
    if total_column:
        totals = pd.to_numeric(frame[total_column].str.strip().str.replace(",", "", regex=False), errors="coerce")
        bad_rows |= ~np.isfinite(totals) | totals.lt(0) | totals.ne(np.floor(totals))
    dates = re.findall(r"(\d{4})\s*년\s*(\d{1,2})\s*월", selected_series)
    period = f"{dates[0][0]}년 {int(dates[0][1])}월" if dates else "헤더에서 기준월 확인 불가"
    return {
        "frame": frame, "names": names, "numeric": numeric, "totals": totals,
        "bad_rows": bad_rows, "ages": ages, "schema": schema, "series": selected_series,
        "series_options": list(series), "encoding": encoding, "region_column": region_column,
        "period": period, "fingerprint": hashlib.sha256(raw).hexdigest(),
    }


def clean_region_name(name):
    return re.sub(r"\s*\(\d+\)\s*$", "", name).strip()


def analyze_region(dataset, index):
    if index not in dataset["names"].index:
        raise ValueError("선택한 지역이 없습니다.")
    name = clean_region_name(dataset["names"].loc[index])
    if dataset["bad_rows"].loc[index]:
        raise ValueError(f"{name}: 빈 셀, 숫자가 아닌 값, 음수 또는 소수 인구수가 있습니다.")
    counts = dataset["numeric"].loc[index].to_numpy(dtype=np.int64)
    ages = np.array(dataset["ages"])
    total = int(counts.sum())
    if total <= 0:
        raise ValueError(f"{name}: 인구가 0명이어서 비율과 평균을 계산할 수 없습니다.")
    if dataset["totals"] is not None and int(dataset["totals"].loc[index]) != total:
        raise ValueError(f"{name}: 총인구와 연령별 합계가 다릅니다. 누락된 연령 또는 계열을 확인하세요.")
    groups = {}
    for label, (start, end) in AGE_GROUPS.items():
        mask = (ages >= start) & (ages <= end if end is not None else True)
        count = int(counts[mask].sum())
        groups[label] = {"count": count, "ratio": count / total * 100}
    open_age = dataset["schema"]["open_age"]
    age_labels = [f"{age}세 이상" if age == open_age else f"{age}세" for age in ages]
    maximum, minimum = int(counts.max()), int(counts.min())
    middle = groups["중년층"]["count"] + groups["장년층"]["count"]
    working = int(counts[(ages >= 15) & (ages <= 64)].sum())
    return {
        "region": name, "source_region": dataset["names"].loc[index], "total": total,
        "mean_age": float(np.dot(ages.astype(float), counts.astype(float)) / total),
        "mean_is_lower_bound": open_age is not None, "open_age": open_age,
        "peak_ages": [label for label, count in zip(age_labels, counts) if count == maximum],
        "least_ages": [label for label, count in zip(age_labels, counts) if count == minimum],
        "peak_count": maximum, "least_count": minimum, "groups": groups,
        "middle_count": middle, "middle_ratio": middle / total * 100,
        "working_count": working, "working_ratio": working / total * 100,
        "aging_ratio": groups["노년층"]["ratio"],
        "aging_level": level(groups["노년층"]["ratio"], AGING_THRESHOLDS),
        "youth_level": level(groups["청년층"]["ratio"], YOUTH_THRESHOLDS),
        "ages": ages.tolist(), "age_labels": age_labels, "counts": counts.tolist(),
        "period": dataset["period"], "series": dataset["series"],
    }


def comparison_table(results):
    return pd.DataFrame([{
        "지역": r["region"], "총인구(명)": r["total"], "평균연령(세)": r["mean_age"],
        "유소년(%)": r["groups"]["유소년층"]["ratio"], "청년(%)": r["groups"]["청년층"]["ratio"],
        "중장년(%)": r["middle_ratio"], "노년/고령화(%)": r["aging_ratio"],
        "최대 연령층": max(r["groups"], key=lambda k: r["groups"][k]["count"]),
    } for r in results])


def automatic_insights(results):
    lines = []
    for r in results:
        largest = max(r["groups"], key=lambda k: r["groups"][k]["count"])
        smallest = min(r["groups"], key=lambda k: r["groups"][k]["count"])
        lines.append(f'{r["region"]}: 최대 단일 연령 구간은 {", ".join(r["peak_ages"])} '
                     f'({r["peak_count"]:,}명), 최대 연령층은 {largest}, 최소 연령층은 {smallest}. '
                     f'청년 {r["groups"]["청년층"]["ratio"]:.2f}%, 노년 {r["aging_ratio"]:.2f}%.')
    if len(results) > 1:
        spreads = {g: max(r["groups"][g]["ratio"] for r in results)
                   - min(r["groups"][g]["ratio"] for r in results) for g in AGE_GROUPS}
        group = max(spreads, key=spreads.get)
        high = max(results, key=lambda r: r["groups"][group]["ratio"])
        low = min(results, key=lambda r: r["groups"][group]["ratio"])
        lines.append(f'지역 간 비율 차이가 가장 큰 연령층은 {group}: '
                     f'{high["region"]}와 {low["region"]}의 차이 {spreads[group]:.2f}%p.')
    return lines


def summary_payload(results, source_name):
    # 원본 CSV와 101개 연령별 행은 보내지 않습니다. 검증된 통계만 전달합니다.
    exclude = {"ages", "age_labels", "counts", "source_region"}
    return {"source": source_name, "regions": [{k: v for k, v in r.items() if k not in exclude}
                                              for r in results],
            "insights": automatic_insights(results),
            "limitations": ["단일 시점의 인구 분포로 인구 증감, 이주, 실제 시설 수요를 확정할 수 없음",
                            "개방 연령 구간은 하한 나이로 계산하여 평균연령이 하한 추정치일 수 있음",
                            "행정구역 부모/자식은 겹치므로 비교 지역을 합산하지 않음",
                            "낮음/보통/높음은 과제 시연용 임의 기준"]}
