"""Streamlit UI. 실행: streamlit run app.py"""
import base64
import hashlib
import io
import json
import re

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import streamlit as st

from ai_analysis import answer_question, demo_analysis, generate_ai_analysis
from config import AGE_GROUPS, BASE_DIR, DATA_PATH, MAX_REGIONS, api_settings
from data_analysis import (analyze_region, automatic_insights, clean_region_name,
                           comparison_table, load_data, summary_payload)
from report_generator import make_report, report_html

st.set_page_config(page_title="AI 인구정보 분석 시스템", page_icon="📊", layout="wide")


def configure_font():
    # Render의 Linux 서버에서도 한글이 보이도록 프로젝트에 폰트를 포함합니다.
    bundled_font = BASE_DIR / "fonts" / "NanumGothic-Regular.ttf"
    if bundled_font.exists():
        font_manager.fontManager.addfont(str(bundled_font))
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"):
        if name in available:
            plt.rcParams["font.family"] = name
            break
    else:
        st.warning("한글 그래프 폰트가 없습니다. 맑은 고딕 또는 나눔고딕을 설치 후 재실행하세요.")
    plt.rcParams["axes.unicode_minus"] = False


@st.cache_data(show_spinner=False, max_entries=4, ttl=3600)
def cached_load(raw, series=None):
    return load_data(raw, series)


def make_charts(results):
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .18, "font.size": 10})
    colors = ["#0891b2", "#f97316", "#7c3aed", "#059669", "#db2777"]
    distribution, ax = plt.subplots(figsize=(11, 4.3), layout="constrained")
    for r, color in zip(results, colors):
        ax.plot(r["ages"], r["counts"], label=r["region"], color=color, linewidth=2)
    ax.set(xlabel="연령 (마지막 개방 구간은 해당 나이 이상)", ylabel="인구수(명)")
    ax.legend(fontsize=9)
    ratios, ax = plt.subplots(figsize=(11, 4.3), layout="constrained")
    x = np.arange(len(AGE_GROUPS))
    width = .8 / len(results)
    for i, (r, color) in enumerate(zip(results, colors)):
        values = [r["groups"][g]["ratio"] for g in AGE_GROUPS]
        bars = ax.bar(x + (i - (len(results) - 1) / 2) * width, values, width,
                      label=r["region"], color=color)
        if len(results) == 1:
            ax.bar_label(bars, fmt="%.1f%%", padding=4)
    ax.set_xticks(x, list(AGE_GROUPS))
    ax.set(ylabel="전체 인구 대비 비율(%)", ylim=(0, max(
        r["groups"][g]["ratio"] for r in results for g in AGE_GROUPS) * 1.2 + 1))
    ax.legend(fontsize=9)
    figures = [("연령별 인구분포", distribution), ("연령대별 인구 비율", ratios)]
    if len(results) > 1:
        comparison, ax = plt.subplots(figsize=(11, 4.3), layout="constrained")
        bottom = np.zeros(len(results))
        for g, color in zip(AGE_GROUPS, colors):
            values = np.array([r["groups"][g]["ratio"] for r in results])
            ax.bar(np.arange(len(results)), values, bottom=bottom, label=g, color=color)
            bottom += values
        ax.set_xticks(np.arange(len(results)), [r["region"].replace(" ", "\n") for r in results])
        ax.set(ylabel="전체 인구 대비 비율(%)", ylim=(0, 105))
        ax.legend(fontsize=9, ncol=5, loc="upper center", bbox_to_anchor=(.5, 1.17))
        figures.append(("지역별 연령층 구성 비교", comparison))
    return figures


def request_cached(cache, key, action):
    # 실패한 응답은 캐시에 남기지 않습니다. 캐시는 현재 사용자 세션에만 존재합니다.
    if key not in cache:
        cache[key] = action()
    return cache[key]


def main():
    configure_font()
    st.title("AI 인구정보 분석 시스템")
    st.write("인구분포 데이터를 기반으로 지역별 인구구조를 분석하고 AI가 분석보고서를 자동으로 작성합니다.")
    st.caption("지역 선택 → 통계·그래프 → AI 해석 → 보고서 저장")
    st.session_state.setdefault("ai_cache", {})
    st.session_state.setdefault("question_cache", {})
    with st.sidebar:
        st.header("분석 설정")
        uploaded = st.file_uploader("CSV 파일 선택 (미선택 시 data/age.csv)", type=["csv"])
        provider = st.selectbox("AI 서비스", ["시연 모드", "ChatGPT", "Gemini"])
        _, default_model = api_settings(provider)
        model = st.text_input("모델 이름", value=default_model, key=f"model_{provider}",
                              disabled=provider == "시연 모드")
        if provider == "시연 모드":
            st.caption("API 없이 Python 통계로 보고서를 만듭니다.")
        elif not api_settings(provider)[0]:
            st.warning(".env에 API KEY를 설정하세요. 통계 분석은 가능합니다.")
    try:
        raw = uploaded.getvalue() if uploaded else DATA_PATH.read_bytes()
        source_name = uploaded.name if uploaded else DATA_PATH.name
        dataset = cached_load(raw)
        with st.sidebar:
            series = st.selectbox("데이터 계열 (월 / 성별)", dataset["series_options"],
                                  index=dataset["series_options"].index(dataset["series"]))
        if series != dataset["series"]:
            dataset = cached_load(raw, series)
    except FileNotFoundError:
        st.error("data/age.csv가 없습니다. 사이드바에서 CSV를 업로드하세요.")
        return
    except (ValueError, OSError) as exc:
        st.error(str(exc))
        return
    st.caption(f'자료 기준: {dataset["period"]} · 계열: {series} · '
               f'{len(dataset["names"]):,}개 지역 · 인코딩: {dataset["encoding"]}')
    if dataset["bad_rows"].any():
        st.warning(f'숫자 형식 오류가 있는 지역 {int(dataset["bad_rows"].sum())}개가 있습니다. 해당 지역은 분석할 수 없습니다.')
    if dataset["schema"]["total_column"] is None:
        st.info("총인구 열을 확인할 수 없어 연령별 인구 합계를 총인구로 사용합니다.")
    if not re.search(r"(?:^|_)(?:총|계|합계|전체)(?:_|$)", series) and series != "연령별 인구":
        st.info("선택한 데이터 계열의 인구를 분석합니다. 남/여 계열은 전체 인구와 다릅니다.")
    with st.expander("데이터 구조와 지표 기준 확인"):
        st.write(f'지역명 열: {dataset["region_column"]} / 총인구 열: {dataset["schema"]["total_column"] or "연령별 합계 사용"}')
        st.write(f'연령 범위: 0~{dataset["ages"][-1]}세 / 마지막 개방 구간: {dataset["schema"]["open_age"]}')
        st.write("고령화: 14% 미만 낮음, 14~20% 미만 보통, 20% 이상 높음. "
                 "청년층: 15% 미만 낮음, 15~25% 미만 보통, 25% 이상 높음. 과제 시연용 임의 기준입니다.")
        st.write("평균연령은 개방 구간을 하한 나이로 계산합니다. 중장년은 30~64세, 생산가능인구는 15~64세입니다.")
        st.dataframe(dataset["frame"].head(3), hide_index=True)
    with st.sidebar:
        query = st.text_input("지역 검색", placeholder="예: 안산, 수원, 성포동")
        # 정규식 검색을 쓰지 않아 [, (, * 등도 일반 문자로 검색합니다.
        mask = dataset["names"].str.contains(query.strip(), regex=False, case=False)
        options = dataset["names"].index[mask].tolist()
        if not options:
            st.info("검색 결과가 없습니다. 다른 검색어를 입력하세요.")
        st.caption(f"검색 결과 {len(options):,}개 · 목록 안에서도 검색 가능")
        selected = st.multiselect("분석지역 선택 (최대 5개)", options=options,
                                  format_func=lambda i: dataset["names"].loc[i],
                                  max_selections=MAX_REGIONS,
                                  key=f'regions_{dataset["fingerprint"]}_{series}_{query}')
        execute = st.button("분석 실행", type="primary", use_container_width=True)
    identity = (dataset["fingerprint"], series, tuple(selected))
    if execute:
        if not selected:
            st.warning("분석할 지역을 1~5개 선택하세요.")
        else:
            try:
                results = [analyze_region(dataset, i) for i in selected]
                st.session_state["active"] = {"identity": identity, "results": results,
                                               "payload": summary_payload(results, source_name)}
                # 이전 지역 보고서와 질문이 새 지역 결과에 섞이지 않도록 초기화합니다.
                st.session_state.pop("report_state", None)
                st.session_state["qa"] = []
            except ValueError as exc:
                st.error(str(exc))
    active = st.session_state.get("active")
    if not active or active["identity"] != identity:
        st.info("사이드바에서 지역을 선택하고 분석 실행을 누르세요. 선택을 바꾸면 다시 분석해야 합니다.")
        return
    results, payload = active["results"], active["payload"]
    st.subheader("1. 주요 지표")
    if len(results) > 1:
        st.info("상위·하위 지역은 인구가 겹칠 수 있습니다. 지역별로 비교하며 선택 지역을 합산하지 않습니다.")
    for r in results:
        st.markdown(f'**{r["region"]}**')
        columns = st.columns(5)
        columns[0].metric("총인구", f'{r["total"]:,}명')
        columns[1].metric("평균연령 (하한 추정)" if r["mean_is_lower_bound"] else "평균연령", f'{r["mean_age"]:.2f}세')
        columns[2].metric("청년층 비율", f'{r["groups"]["청년층"]["ratio"]:.2f}%')
        columns[3].metric("노년층/고령화 비율", f'{r["aging_ratio"]:.2f}%')
        columns[4].metric("고령화 정도 (임의 기준)", r["aging_level"])
        st.caption(f'청년층 정도: {r["youth_level"]} · 생산가능인구: {r["working_count"]:,}명 '
                   f'({r["working_ratio"]:.2f}%) · 중장년: {r["middle_count"]:,}명 ({r["middle_ratio"]:.2f}%)')
    for insight in automatic_insights(results):
        st.write("• " + insight)
    chart_images = []
    for index, (label, figure) in enumerate(make_charts(results), start=2):
        st.subheader(f"{index}. {label}")
        st.pyplot(figure)
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
        chart_images.append((label, base64.b64encode(buffer.getvalue()).decode("ascii")))
        plt.close(figure)
    st.subheader("지역별 통계표")
    table = comparison_table(results)
    st.dataframe(table.style.format({c: "{:,.2f}" for c in table.columns
                                    if c not in ("지역", "최대 연령층", "총인구(명)")}), hide_index=True)
    detailed = pd.DataFrame([{"지역": r["region"], "연령층": g, "인구수(명)": v["count"],
                              "비율(%)": round(v["ratio"], 2)} for r in results for g, v in r["groups"].items()])
    st.dataframe(detailed, hide_index=True)
    st.download_button("통계표 CSV 저장", table.to_csv(index=False).encode("utf-8-sig"),
                       "population_comparison.csv", "text/csv")
    st.subheader("5. AI 분석")
    st.caption("버튼을 누를 때만 호출합니다. 동일 통계·서비스·모델의 성공 응답은 현재 세션에서 재사용합니다.")
    request_key = hashlib.sha256(json.dumps([payload, provider, model], ensure_ascii=False,
                                           sort_keys=True).encode("utf-8")).hexdigest()
    if st.button("기본 보고서 생성" if provider == "시연 모드" else "AI 분석 및 보고서 생성", type="primary"):
        try:
            with st.spinner("보고서를 작성하고 있습니다…"):
                analysis = request_cached(st.session_state["ai_cache"], request_key,
                    lambda: demo_analysis(payload) if provider == "시연 모드"
                    else generate_ai_analysis(payload, provider, model))
                report = make_report(results, analysis, source_name, provider, model)
                st.session_state["report_state"] = {"key": request_key, "analysis": analysis, "report": report}
        except ValueError as exc:
            st.error(str(exc))
    report_state = st.session_state.get("report_state")
    if report_state and report_state["key"] == request_key:
        st.info("Python 계산 기반 시연 결과입니다. AI는 호출하지 않았습니다." if provider == "시연 모드"
                else f"{provider} / {model} 분석 결과입니다. 수치 근거와 해석을 확인하세요.")
        st.markdown(report_state["analysis"]["findings"])
        st.subheader("6. 인구정보 분석보고서")
        report = report_state["report"]
        with st.expander("전체 보고서 보기", expanded=True):
            st.markdown(report)
        html = report_html(report, chart_images)
        columns = st.columns(3)
        columns[0].download_button("Markdown 저장", report.encode("utf-8-sig"), "population_report.md", "text/markdown")
        columns[1].download_button("TXT 저장", report.encode("utf-8-sig"), "population_report.txt", "text/plain")
        columns[2].download_button("HTML 저장 (그래프 포함)", html.encode("utf-8"), "population_report.html", "text/html")
        st.caption("PDF 선택 기능: 저장한 HTML을 Edge/Chrome에서 열고 Ctrl+P → PDF로 저장하세요. 한글과 그래프를 함께 저장할 수 있습니다.")
    elif report_state:
        st.info("서비스 또는 모델이 바뀌었습니다. 해당 설정으로 보고서를 생성하세요.")
    st.subheader("7. AI 추가 질문")
    if provider == "시연 모드":
        st.info("추가 질문은 ChatGPT 또는 Gemini를 선택하고 API KEY를 설정하면 사용할 수 있습니다.")
    elif not report_state or report_state["key"] != request_key:
        st.info("먼저 현재 서비스로 분석보고서를 생성하세요.")
    else:
        with st.form("question_form", clear_on_submit=True):
            question = st.text_area("현재 지역 데이터에 질문하기", max_chars=1000,
                                    placeholder="노년층 비율이 높은 지역의 복지 서비스는 무엇을 검토할 수 있나요?")
            submit = st.form_submit_button("질문 보내기")
        if submit:
            try:
                qkey = hashlib.sha256((request_key + question.strip()).encode("utf-8")).hexdigest()
                with st.spinner("AI 답변을 기다리고 있습니다…"):
                    answer = request_cached(st.session_state["question_cache"], qkey,
                                            lambda: answer_question(payload, question, provider, model))
                st.session_state.setdefault("qa", []).append((request_key, question, answer))
            except ValueError as exc:
                st.error(str(exc))
        for key, question, answer in st.session_state.get("qa", [])[-10:]:
            if key == request_key:
                with st.chat_message("user"):
                    st.write(question)
                with st.chat_message("assistant"):
                    st.markdown(answer)


if __name__ == "__main__":
    main()
