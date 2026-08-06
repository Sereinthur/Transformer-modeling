import unittest

from transformer_modeling import Config, estimate
from transformer_modeling.config import Config as DomainConfig
from transformer_modeling.models import resolve_model_definition
from transformer_modeling.operators import get_operator_catalog

from helpers import example


class PublicInterfaceTests(unittest.TestCase):
    def test_top_level_api_is_stable_and_v3_only(self):
        self.assertIs(Config, DomainConfig)
        self.assertTrue(callable(estimate))
        self.assertEqual(estimate(Config.from_dict(example()), False)["schema_version"], 3)

    def test_new_public_catalog_and_resolver(self):
        catalog = get_operator_catalog()
        self.assertEqual(catalog["schema_version"], 3)
        self.assertIn("standard_attention", {item["type"] for item in catalog["operators"]})
        resolved = resolve_model_definition(preset_id="qwen3-30b-a3b")
        self.assertIn("moe", [item["operator"]["type"] for item in resolved["resolved_model"]["layer_pattern"][0]["operations"]])

    def test_huggingface_config_maps_to_ordered_operators(self):
        resolved = resolve_model_definition(hf_config={
            "model_type": "llama", "_name_or_path": "demo",
            "num_hidden_layers": 2, "hidden_size": 128,
            "intermediate_size": 256, "vocab_size": 1024,
            "num_attention_heads": 4, "num_key_value_heads": 2,
        })["resolved_model"]
        self.assertEqual(resolved["dimensions"]["layer_count"], 2)
        self.assertIn("gated_ffn", [item["operator"]["type"] for item in resolved["layer_pattern"][0]["operations"]])


if __name__ == "__main__":
    unittest.main()
