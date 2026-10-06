"""정확한 통계 표와 AI 해석을 결합하고 Markdown/TXT/HTML을 만듭니다."""
from datetime import datetime
from html import escape
from zoneinfo import ZoneInfo

import bleach
import markdown


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def make_report(results, analysis, source_name, provider, model):
    time = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M KST")
    names = ", ".join(r["region"] for r in results)
    basic = ["| 지역 | 총인구(명) | 평균연령(세) | 청년(%) | 노년/고령화(%) |",
             "|---|---:|---:|---:|---:|"]
    group_table = ["| 지역 | 연령층 | 인구수(명) | 비율(%) |", "|---|---|---:|---:|"]
    for r in results:
        basic.append(f'| {cell(r["region"])} | {r["total"]:,} | {r["mean_age"]:.2f} | '
                     f'{r["groups"]["청년층"]["ratio"]:.2f} | {r["aging_ratio"]:.2f} |')
        for group, value in r["groups"].items():
            group_table.append(f'| {cell(r["region"])} | {group} | {value["count"]:,} | {value["ratio"]:.2f} |')
    mode = "Python 통계 기반 기본 보고서 (AI 미호출)" if provider == "시연 모드" else f"{provider} / {model}"
    notes = "\n".join([
        "- 평균연령: Σ(연령 × 해당 인구) / 연령별 합계. '100세 이상' 등은 하한 나이를 사용하므로 하한 추정치입니다.",
        "- 유소년 0~14세, 청년 15~29세, 중년 30~49세, 장년 50~64세, 노년 65세 이상.",
        "- 중장년은 30~64세, 생산가능인구는 15~64세. 고령화 비율은 노년층 비율과 같습니다.",
        "- 고령화 지표: 14% 미만 낮음, 14~20% 미만 보통, 20% 이상 높음 (시연용 임의 기준).",
        "- 청년 지표: 15% 미만 낮음, 15~25% 미만 보통, 25% 이상 높음 (시연용 임의 기준).",
        "- 상위·하위 행정구역은 인구가 겹칠 수 있습니다. 선택 지역의 인구를 합산하지 않았습니다.",
        "- 자료 시점과 작성 시점은 다릅니다. 인구 증감·미래 추세·실제 시설 부족은 이 자료만으로 확인할 수 없습니다.",
        "- AI 해석과 정책 제안은 별도 검토가 필요합니다. 통계표는 Python에서 계산했습니다.",
    ])
    parts = ["# 인구정보 분석보고서", "## 1. 분석 개요",
             f"- 목적: 지역별 인구구조 파악 및 선택 지역 비교\n- 분석 대상: {names}\n"
             f'- 사용 데이터: {source_name}\n- 데이터 기준: {results[0]["period"]}\n'
             f'- 데이터 계열: {results[0]["series"]}\n- 작성 시각: {time}\n- 생성 방식: {mode}',
             "### 계산 기준과 한계", notes, "## 2. 지역 기본 정보", "\n".join(basic),
             "## 3. 연령별 인구분포 분석", analysis["distribution"],
             "## 4. 주요 연령층 분석", "\n".join(group_table), analysis["groups"]]
    if len(results) > 1:
        parts += ["## 5. 지역 비교분석", analysis["comparison"]]
    parts += ["## 6. AI 분석 결과" if provider != "시연 모드" else "## 6. Python 자동 인사이트 (AI 미호출)",
              analysis["findings"], "## 7. 정책 및 지역 서비스 제안", analysis["policies"],
              "## 8. 최종 결론", analysis["conclusion"]]
    return "\n\n".join(parts)


def report_html(report, chart_images=()):
    # AI 응답과 CSV는 신뢰할 수 없는 입력이므로 HTML 태그와 링크를 제한합니다.
    body = markdown.markdown(report, extensions=["tables"])
    body = bleach.clean(body, tags=["h1", "h2", "h3", "h4", "p", "br", "strong", "em",
                                   "ul", "ol", "li", "blockquote", "code", "pre",
                                   "table", "thead", "tbody", "tr", "th", "td"],
                        attributes={}, strip=True)
    charts = "".join(f'<figure><img alt="{escape(label)}" src="data:image/png;base64,{encoded}"></figure>'
                     for label, encoded in chart_images)
    return f'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>인구정보 분석보고서</title>
<style>body{{font-family:"Malgun Gothic",sans-serif;max-width:1050px;margin:40px auto;padding:24px;color:#172033;line-height:1.7}}
h1,h2{{color:#155e75}}table{{border-collapse:collapse;width:100%;font-size:14px}}
td,th{{border:1px solid #ccd5de;padding:8px;text-align:left}}th{{background:#eef6f8}}
img{{max-width:100%}}figure{{margin:24px 0}}@media print{{body{{margin:0;padding:8px}}h2{{break-after:avoid}}figure{{break-inside:avoid}}}}
</style></head><body>{body}<h2>부록: 분석 그래프</h2>{charts}</body></html>'''
