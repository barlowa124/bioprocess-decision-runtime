"""Shared provenance helper (docs/PROVENANCE.md, schema v1)."""
import tempfile
import unittest
from pathlib import Path

from bioprocess_runtime import provenance_conv as prov


class ProvenanceConvTests(unittest.TestCase):
    def test_manifest_shape(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "in.txt"
            f.write_text("data")
            m = prov.manifest("test-tool", inputs=[f], config={"a": 1})
            self.assertEqual(m["schema"], "provenance/v1")
            self.assertEqual(m["input_sha256"][str(f)], prov.sha256_file(f))
            self.assertEqual(m["config_sha256"], prov.sha256_obj({"a": 1}))
            self.assertIn("sha", m["git"])

    def test_canonical_config_hash_ignores_key_order(self):
        self.assertEqual(prov.sha256_obj({"b": 2, "a": 1}),
                         prov.sha256_obj({"a": 1, "b": 2}))

    def test_verify_catches_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "out.txt"
            f.write_text("v1")
            m = prov.manifest("t")
            prov.seal(m, [f])
            self.assertEqual(prov.verify(m), [])
            f.write_text("v2")
            self.assertIn("sha256 mismatch", prov.verify(m)[0])

    def test_missing_input_omitted(self):
        self.assertEqual(
            prov.manifest("t", inputs=["/nonexistent"])["input_sha256"], {})


if __name__ == "__main__":
    unittest.main()
