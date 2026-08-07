import tempfile
import unittest
from pathlib import Path

from src.utils.utils import replace_tire_force_tables


class TireTableReplacementTests(unittest.TestCase):
    def _write_component(self, root, component, axis_value=1.0):
        keyword = {
            "Fx": "FX_TIRE_CARPET",
            "Fy": "FY_TIRE_CARPET",
            "Mz": "MZ_TIRE_CARPET",
        }[component]
        par_path = root / f"{component}.par"
        par_path.write_text(
            "PARSFILE\n"
            f"{keyword} 2D_FROM_ZERO\n"
            "0, 100, 200\n"
            f"{axis_value}, 10, 20\n"
            "ENDTABLE\n"
            "LOG_ENTRY unchanged\n",
            encoding="utf-8",
        )
        return par_path

    def _write_csv(self, root, component, axis_value=1.0):
        csv_path = root / f"{component}.csv"
        csv_path.write_text(
            "0,100,200\n"
            f"{axis_value},30,40\n",
            encoding="utf-8",
        )

    def test_replaces_all_three_tables_and_creates_backups(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            table_dir = root / "tables"
            table_dir.mkdir()
            targets = {}
            for component in ("Fx", "Fy", "Mz"):
                targets[component] = self._write_component(root, component)
                self._write_csv(table_dir, component)

            written = replace_tire_force_tables(table_dir, targets)

            self.assertEqual(set(written), set(targets.values()))
            for par_path in targets.values():
                updated = par_path.read_text(encoding="utf-8")
                self.assertIn("1.0, 30, 40", updated)
                self.assertIn("LOG_ENTRY unchanged", updated)
                backup = par_path.with_suffix(
                    par_path.suffix + ".pre_randomization.bak"
                )
                self.assertTrue(backup.is_file())
                self.assertIn("1.0, 10, 20", backup.read_text(encoding="utf-8"))

    def test_rejects_axis_mismatch_before_writing_any_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            table_dir = root / "tables"
            table_dir.mkdir()
            targets = {}
            originals = {}
            for component in ("Fx", "Fy", "Mz"):
                targets[component] = self._write_component(root, component)
                originals[component] = targets[component].read_text(
                    encoding="utf-8"
                )
                mismatch = 2.0 if component == "Mz" else 1.0
                self._write_csv(table_dir, component, axis_value=mismatch)

            with self.assertRaisesRegex(ValueError, "Slip axis mismatch"):
                replace_tire_force_tables(table_dir, targets)

            for component, par_path in targets.items():
                self.assertEqual(
                    par_path.read_text(encoding="utf-8"), originals[component]
                )


if __name__ == "__main__":
    unittest.main()
