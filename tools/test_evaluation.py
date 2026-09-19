import json
import tempfile
import unittest
from pathlib import Path

import monitor


class EvaluationTest(unittest.TestCase):

    def test_dashboard_reads_ladder(self):
        with tempfile.TemporaryDirectory() as root:
            run = Path(root) / "sample"
            run.mkdir()
            (run / "log.json").write_text(json.dumps({
                "cfg": {}, "snapshots": []}))
            report = {"complete": True, "pairs": [{"games": 2}]}
            (run / "ladder.json").write_text(json.dumps(report))
            comparisons = run / "comparisons"
            comparisons.mkdir()
            (comparisons / "baseline.json").write_text(json.dumps(report))
            detail = monitor.detail(root, "sample")
            self.assertTrue(detail["ladder"]["complete"])
            self.assertEqual(detail["ladder"]["pairs"][0]["games"], 2)
            self.assertEqual(detail["comparisons"][0]["pairs"][0]["games"], 2)


if __name__ == "__main__":
    unittest.main()
