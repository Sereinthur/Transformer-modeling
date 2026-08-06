import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ArchitectureTests(unittest.TestCase):
    def test_old_family_estimators_are_removed(self):
        removed = (
            "core/prefill.py", "core/decode.py", "core/moe.py", "core/hybrid_work.py",
            "estimators/single.py", "estimators/kimi_k3.py",
            "parallel/tensor/estimator.py", "parallel/pipeline/estimator.py",
        )
        for relative in removed:
            self.assertFalse((ROOT / "transformer_modeling" / relative).exists(), relative)

    def test_estimator_has_no_model_family_branch(self):
        estimator = (ROOT / "transformer_modeling" / "estimators" / "composed.py").read_text(encoding="utf-8")
        for marker in ("is_kimi", "uses_hybrid", "family ==", "kimi_k3"):
            self.assertNotIn(marker, estimator)

    def test_domain_modules_stay_focused(self):
        for path in (ROOT / "transformer_modeling").rglob("*.py"):
            if path.parent == ROOT / "transformer_modeling":
                continue
            if "visual_app" in path.parts:
                continue
            self.assertLessEqual(len(path.read_text(encoding="utf-8").splitlines()), 320, path)

    def test_browser_entry_and_modules(self):
        calibration = ROOT / "transformer_modeling" / "calibration"
        self.assertFalse(any(calibration.glob("*.py")))
        static = ROOT / "transformer_modeling" / "visual_app" / "static"
        entry = (static / "index.html").read_text(encoding="utf-8")
        self.assertIn('js/app.js', entry)
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node.js unavailable")
        for path in (static / "js").glob("*.js"):
            completed = subprocess.run([node, "--input-type=module", "--check"], input=path.read_bytes(), capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr.decode(errors="replace"))


if __name__ == "__main__":
    unittest.main()
