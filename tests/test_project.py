"""unittest로 실행합니다. API는 모의 응답으로 비용 없이 검증합니다."""
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_analysis import load_data, analyze_region, summary_payload, comparison_table
from ai_analysis import demo_analysis, generate_ai_analysis, call_ai
from report_generator import make_report, report_html
from config import DATA_PATH, level, AGING_THRESHOLDS


def sample(counts=None, total=None, encoding="utf-8-sig", prefix=""):
    import csv
    counts = counts if counts is not None else [1] * 101
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["지역명", prefix + "총인구수"] + [prefix + f"{a}세" + (" 이상" if a == 100 else "") for a in range(101)])
    writer.writerow(["테스트시 테스트동(123)", sum(counts) if total is None else total] + counts)
    return output.getvalue().encode(encoding)


class AnalysisTests(unittest.TestCase):
    def test_actual_csv(self):
        ds = load_data(DATA_PATH.read_bytes())
        self.assertEqual(len(ds["names"]), 3827)
        self.assertEqual(ds["ages"], list(range(101)))
        self.assertEqual(ds["encoding"], "cp949")
        self.assertEqual(ds["period"], "2020년 9월")
        result = analyze_region(ds, 0)
        self.assertEqual(result["total"], 9699232)
        self.assertEqual(result["region"], "서울특별시")
        self.assertAlmostEqual(sum(g["ratio"] for g in result["groups"].values()), 100)
        # 실제 데이터 모든 행의 합계와 숫자 형식을 확인합니다.
        for index in ds["names"].index:
            r = analyze_region(ds, index)
            self.assertEqual(sum(g["count"] for g in r["groups"].values()), r["total"])

    def test_boundaries_mean_and_ties(self):
        ds = load_data(sample())
        r = analyze_region(ds, 0)
        self.assertEqual([g["count"] for g in r["groups"].values()], [15, 15, 20, 15, 36])
        self.assertEqual(r["working_count"], 50)
        self.assertEqual(r["middle_count"], 35)
        self.assertEqual(r["mean_age"], 50)
        self.assertTrue(r["mean_is_lower_bound"])
        self.assertEqual(len(r["peak_ages"]), 101)
        self.assertEqual(level(14, AGING_THRESHOLDS), "보통")
        self.assertEqual(level(20, AGING_THRESHOLDS), "높음")

    def test_encodings_and_comma(self):
        for encoding in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
            r = analyze_region(load_data(sample(encoding=encoding)), 0)
            self.assertEqual(r["total"], 101)
        import csv
        rows = list(csv.reader(io.StringIO(sample([1000] * 101).decode("utf-8-sig"))))
        rows[1][1:] = [f"{int(v):,}" for v in rows[1][1:]]
        output = io.StringIO()
        csv.writer(output).writerows(rows)
        raw = output.getvalue().encode("utf-8")
        self.assertEqual(analyze_region(load_data(raw), 0)["total"], 101000)

    def test_bad_values_and_zero(self):
        for value in ("", "abc", "-1", "1.5", "inf"):
            raw = sample().decode("utf-8-sig")
            rows = raw.splitlines()
            cells = rows[1].split(",")
            cells[2] = value
            rows[1] = ",".join(cells)
            with self.assertRaises(ValueError):
                analyze_region(load_data("\n".join(rows).encode("utf-8")), 0)
        with self.assertRaises(ValueError):
            analyze_region(load_data(sample([0] * 101)), 0)
        with self.assertRaises(ValueError):
            analyze_region(load_data(sample(total=102)), 0)

    def test_bad_schema(self):
        for raw in (b"", b"a,b\n", b"a,b\nx,y", b"\xff\xfe\xfa", b"region,0\nx,1"):
            with self.assertRaises(ValueError):
                load_data(raw)
        # 중간 연령이 누락된 CSV를 거부합니다.
        raw = sample().decode("utf-8-sig").replace("50세", "unknown")
        with self.assertRaises(ValueError):
            load_data(raw.encode("utf-8"))

    def test_multiple_series_and_summary(self):
        import pandas as pd
        one = pd.read_csv(io.BytesIO(sample(prefix="2020년09월_총_")))
        two = pd.read_csv(io.BytesIO(sample(prefix="2020년09월_남_")))
        frame = pd.concat([one, two.drop(columns="지역명")], axis=1)
        raw = frame.to_csv(index=False).encode("utf-8")
        ds = load_data(raw)
        self.assertEqual(len(ds["series_options"]), 2)
        self.assertEqual(ds["series"], "2020년09월_총")
        r = analyze_region(load_data(raw, "2020년09월_남"), 0)
        payload = summary_payload([r], "test.csv")
        self.assertNotIn("counts", payload["regions"][0])
        self.assertEqual(len(comparison_table([r])), 1)


class ReportAndAITests(unittest.TestCase):
    def setUp(self):
        self.result = analyze_region(load_data(sample()), 0)
        self.payload = summary_payload([self.result], "test.csv")

    def test_report_and_sanitization(self):
        report = make_report([self.result], demo_analysis(self.payload), "test.csv", "시연 모드", "local")
        self.assertIn("AI 미호출", report)
        self.assertIn("## 8. 최종 결론", report)
        self.assertNotIn("## 5. 지역 비교분석", report)
        html = report_html(report + '\n<script>alert(1)</script><img src=x onerror="attack()">')
        self.assertNotIn("<script", html)
        self.assertNotIn("onerror", html)
        self.assertIn("<table>", html)

    def test_ai_json_and_failure(self):
        data = demo_analysis(self.payload)
        with patch("ai_analysis.call_ai", return_value=json.dumps(data)):
            self.assertEqual(generate_ai_analysis(self.payload, "ChatGPT", "test"), data)
        for response in ("", "invalid", "[]", "{}", '{"findings": 1}'):
            with patch("ai_analysis.call_ai", return_value=response):
                with self.assertRaises(ValueError):
                    generate_ai_analysis(self.payload, "ChatGPT", "test")
        with patch("ai_analysis.api_settings", return_value=("", "test")):
            with self.assertRaises(ValueError):
                call_ai("ChatGPT", "test", "hi")

    def test_openai_adapter(self):
        import openai
        client = MagicMock()
        client.__enter__.return_value = client
        client.responses.create.return_value.status = "completed"
        client.responses.create.return_value.output_text = '{"ok": true}'
        with patch("ai_analysis.api_settings", return_value=("test-key", "test")), patch.object(openai, "OpenAI", return_value=client):
            self.assertEqual(call_ai("ChatGPT", "test", "hi", True), '{"ok": true}')
            self.assertFalse(client.responses.create.call_args.kwargs["store"])

    def test_gemini_adapter(self):
        from google import genai
        from google.genai import types
        # 실제 SDK 타입을 생성해 설정 항목의 호환성도 확인합니다.
        types.HttpOptions(timeout=90000, retry_options=types.HttpRetryOptions(attempts=1))
        client = MagicMock()
        client.__enter__.return_value = client
        response = client.models.generate_content.return_value
        response.text = "answer"
        response.candidates = [MagicMock(finish_reason=types.FinishReason.STOP)]
        with patch("ai_analysis.api_settings", return_value=("test-key", "test")), patch.object(genai, "Client", return_value=client):
            self.assertEqual(call_ai("Gemini", "test", "hi"), "answer")


class UITests(unittest.TestCase):
    def test_streamlit_demo_and_invalid_search(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60).run()
        self.assertEqual(len(at.exception), 0)
        at.sidebar.multiselect[0].set_value([0, 1])
        at.sidebar.button[0].click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.metric), 10)
        report_button = next(b for b in at.button if b.label == "기본 보고서 생성")
        report_button.click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertIn("report_state", at.session_state)
        self.assertIn("## 5. 지역 비교분석", at.session_state["report_state"]["report"])
        search = next(t for t in at.sidebar.text_input if t.label == "지역 검색")
        search.set_value("존재하지않는지역[").run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.metric), 0)


if __name__ == "__main__":
    unittest.main()
