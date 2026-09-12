from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from collect_data.relativize_saved_paths import relativize_json_file


class RelativizeSavedPathsTests(unittest.TestCase):
    def test_rewrites_nested_repository_paths_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary) / "project"
            record = repository / "scan_output/run_3/report.json"
            record.parent.mkdir(parents=True)
            record.write_text(
                json.dumps(
                    {
                        "manifest": str(repository / "scan_output/run_3/manifest.json"),
                        "configuration": str(repository / "scripts/config.yaml"),
                        "nested": [str(repository / "scan_output/run_3/masks/a.png")],
                        "already_relative": "masks/b.png",
                        "isaac_prim": "/World/YCBObjects/object_1",
                        "external": "/opt/ros/humble/setup.bash",
                    }
                ),
                encoding="utf-8",
            )

            count = relativize_json_file(
                record, repository.resolve(), dry_run=False, backup=False
            )
            updated = json.loads(record.read_text(encoding="utf-8"))

            self.assertEqual(count, 3)
            self.assertEqual(updated["manifest"], "manifest.json")
            self.assertEqual(updated["configuration"], "../../scripts/config.yaml")
            self.assertEqual(updated["nested"], ["masks/a.png"])
            self.assertEqual(updated["already_relative"], "masks/b.png")
            self.assertEqual(updated["isaac_prim"], "/World/YCBObjects/object_1")
            self.assertEqual(updated["external"], "/opt/ros/humble/setup.bash")

    def test_dry_run_does_not_modify_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary) / "project"
            record = repository / "scan_output/run_1/manifest.json"
            record.parent.mkdir(parents=True)
            original = json.dumps({"path": str(repository / "scripts/config.yaml")})
            record.write_text(original, encoding="utf-8")

            count = relativize_json_file(
                record, repository.resolve(), dry_run=True, backup=False
            )

            self.assertEqual(count, 1)
            self.assertEqual(record.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
