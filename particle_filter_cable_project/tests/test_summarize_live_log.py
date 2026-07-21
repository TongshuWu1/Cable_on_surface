import unittest

from tools.summarize_live_log import parse_tracking_line


class LiveLogSummaryTests(unittest.TestCase):
    def test_union_selector_metrics_and_stage_are_parsed(self):
        line = (
            "Frame 42 | GUI 20.0 FPS | CAPTURE 60.0 FPS | TRACK 15.0 FPS "
            "| cables:pidnet | union-rank=12,1 rms=15.3mm cov=0.92 "
            "gain=+2.2mm/+0.03cov | pfms:prep=0.7,union=6.3,est=7.4"
        )

        record = parse_tracking_line(line)

        self.assertIsNotNone(record)
        self.assertAlmostEqual(record["union_coverage_rms_mm"], 15.3)
        self.assertAlmostEqual(record["union_covered_fraction"], 0.92)
        self.assertAlmostEqual(record["union_rms_gain_mm"], 2.2)
        self.assertAlmostEqual(record["union_coverage_fraction_gain"], 0.03)
        self.assertAlmostEqual(record["pf_union_ms"], 6.3)


if __name__ == "__main__":
    unittest.main()
