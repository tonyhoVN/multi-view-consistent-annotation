from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

ANNOTATION_DIR = (
    Path(__file__).resolve().parents[1] / "scripts/annotation_propagation"
)
if str(ANNOTATION_DIR) not in sys.path:
    sys.path.insert(0, str(ANNOTATION_DIR))

from evaluate_segmentation_map import first_view_classes


class EvaluationObjectSelectionTests(unittest.TestCase):
    def test_classes_come_only_from_first_route_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_directory = Path(temporary) / "run_29"
            segmentation = run_directory / "save_segment/segment_0/manifest.json"
            segmentation.parent.mkdir(parents=True)
            segmentation.write_text(
                json.dumps(
                    {
                        "objects": [
                            {"instance": "object_001_003_cracker_box"},
                            {"instance": "object_002_011_banana"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            scan_manifest = run_directory / "manifest.json"
            captures = [
                {
                    "sample_index": 0,
                    "segmentation": {
                        "status": "saved",
                        "manifest": "save_segment/segment_0/manifest.json",
                    },
                }
            ]

            self.assertEqual(
                first_view_classes(scan_manifest, captures),
                {"cracker_box", "banana"},
            )


if __name__ == "__main__":
    unittest.main()
