from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from collect_data.migrate_scan_layout import migrate_run
from scan_layout import ScanRunLayout, resolve_saved_path, serialized_relative_path


class ScanLayoutTests(unittest.TestCase):
    def test_missing_legacy_repository_path_rebases_to_current_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary) / "new_checkout"
            configuration = repository / "scripts/path_planning_single/config.yaml"
            configuration.parent.mkdir(parents=True)
            configuration.write_text("config", encoding="utf-8")

            resolved = resolve_saved_path(
                "/old/machine/project/scripts/path_planning_single/config.yaml",
                repository / "scan_output/run_36",
                repository,
            )

            self.assertEqual(resolved, configuration)

    def test_serialized_path_is_relative_to_record_directory(self) -> None:
        self.assertEqual(
            serialized_relative_path(
                Path("/tmp/project/scripts/config.yaml"),
                Path("/tmp/project/scan_output/run_3"),
            ),
            "../../scripts/config.yaml",
        )

    def test_layout_places_every_artifact_under_run(self) -> None:
        layout = ScanRunLayout(Path("/tmp/scans"), "run_7")
        self.assertEqual(layout.manifest, Path("/tmp/scans/run_7/manifest.json"))
        self.assertEqual(layout.images, Path("/tmp/scans/run_7/save_images"))
        self.assertEqual(
            layout.baseline_segment("naive_vlm", "multi-shot"),
            Path(
                "/tmp/scans/run_7/baseline_segment/naive_vlm_multi_shot"
            ),
        )

    def test_migration_moves_files_and_rewrites_manifest_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "save_images_run_3"
            images.mkdir()
            (images / "color_0.png").write_bytes(b"image")
            old_manifest = root / "manifest_run_3.json"
            old_manifest.write_text(
                json.dumps(
                    {
                        "output_suffix": "run_3",
                        "directories": {"images": "save_images_run_3"},
                        "captures": [{"color_image": "save_images_run_3/color_0.png"}],
                    }
                ),
                encoding="utf-8",
            )

            migrate_run(root, "run_3", dry_run=False)

            manifest_path = root / "run_3" / "manifest.json"
            document = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(document["directories"]["images"], "save_images")
            self.assertEqual(
                document["captures"][0]["color_image"], "save_images/color_0.png"
            )
            self.assertTrue((root / "run_3/save_images/color_0.png").is_file())
            self.assertFalse(old_manifest.exists())

    def test_migration_refuses_destination_collision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "save_TF_run_2").mkdir()
            (root / "run_2/save_TF").mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                migrate_run(root, "run_2", dry_run=False)


if __name__ == "__main__":
    unittest.main()
