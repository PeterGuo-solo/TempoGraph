import json
import tempfile
import unittest
from pathlib import Path

from tempograph.config import load_config


class ConfigTest(unittest.TestCase):
    def test_base_config_is_consistent(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "base.json"
        config = load_config(path)
        self.assertEqual(config["motion"]["feature_dim"], 139)
        self.assertEqual(config["experiment"]["training_seeds"], [1234, 2025])
        self.assertEqual(config["experiment"]["split_seed"], 1234)

    def test_bad_feature_dimension_is_rejected(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "base.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        config["motion"]["feature_dim"] = 140
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "bad.json"
            bad.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(bad)


if __name__ == "__main__":
    unittest.main()
