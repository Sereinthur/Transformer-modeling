import unittest
from copy import deepcopy

from transformer_modeling import Config, estimate
from transformer_modeling.config import Config as DomainConfig, expand_preset_config
from transformer_modeling.models import resolve_model_definition
from transformer_modeling.operators import get_operator_catalog
from transformer_modeling.visual_app.flowchart_schema import config_to_flowchart

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

    def test_preset_backed_config_expands_model_without_mutating_input(self):
        data = example("moe_qwen3_30b_a3b")
        original = deepcopy(data)
        config = Config.from_dict(data)
        self.assertEqual(config.model.model_id, "qwen3-30b-a3b")
        self.assertEqual(data, original)

    def test_preset_backed_config_rejects_embedded_model(self):
        data = example("moe_qwen3_30b_a3b")
        data["model"] = resolve_model_definition(preset_id="qwen3-30b-a3b")["resolved_model"]
        with self.assertRaisesRegex(ValueError, "either preset_id or model"):
            Config.from_dict(data)

    def test_preset_expansion_produces_a_standalone_editable_config(self):
        expanded = expand_preset_config(example("moe_qwen3_30b_a3b"))
        self.assertNotIn("preset_id", expanded)
        self.assertEqual(expanded["model"]["id"], "qwen3-30b-a3b")
        self.assertEqual(config_to_flowchart(expanded)["model_info"]["hidden_size"], 2048)

    def test_huggingface_config_maps_to_ordered_operators(self):
        resolved = resolve_model_definition(hf_config={
            "model_type": "llama", "_name_or_path": "demo",
            "num_hidden_layers": 2, "hidden_size": 128,
            "intermediate_size": 256, "vocab_size": 1024,
            "num_attention_heads": 4, "num_key_value_heads": 2,
        })["resolved_model"]
        self.assertEqual(resolved["dimensions"]["layer_count"], 2)
        self.assertIn("gated_ffn", [item["operator"]["type"] for item in resolved["layer_pattern"][0]["operations"]])

    def test_huggingface_config_rejects_implicit_non_divisible_head_width(self):
        with self.assertRaisesRegex(ValueError, "hidden_size must be divisible"):
            resolve_model_definition(hf_config={
                "num_hidden_layers": 2, "hidden_size": 1000, "intermediate_size": 256,
                "vocab_size": 1024, "num_attention_heads": 3,
            })

    def test_unknown_preset_scenario_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "only scenario='base'"):
            resolve_model_definition(preset_id="qwen3-30b-a3b", scenario="performance")


if __name__ == "__main__":
    unittest.main()
