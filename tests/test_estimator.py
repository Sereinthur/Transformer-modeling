import copy
import unittest

from transformer_modeling import Config, estimate
from transformer_modeling.models import get_preset

from helpers import example, phase_operators


class EstimatorTests(unittest.TestCase):
    def test_schema_v2_is_required(self):
        data = example()
        data["schema_version"] = 1
        with self.assertRaisesRegex(ValueError, "expected 2"):
            Config.from_dict(data)

    def test_fixed_skeleton_expands_cyclic_pattern(self):
        data = example()
        data["model"] = get_preset("kimi-k3-draft", "compact")["model"]
        data["hardware"]["compute"]["throughput"] = {"mxfp4_mxfp8_ops_per_second": 1e15}
        config = Config.from_dict(data)
        types = [layer.attention.type for layer in config.model.expanded_layers()]
        self.assertEqual(types[:8], ["kda", "kda", "kda", "gated_mla"] * 2)

    def test_standard_model_is_sum_of_registered_operators(self):
        result = estimate(Config.from_dict(example()), details=True)
        operators = phase_operators(result["performance"]["prefill"])
        self.assertEqual(
            {item["type"] for item in operators},
            {"token_embedding", "rms_norm", "standard_attention", "standard_residual", "gated_ffn", "lm_head", "sampling"},
        )
        self.assertEqual(result["model"]["parameters"], 5_802_037_248)

    def test_decode_latency_grows_with_context(self):
        result = estimate(Config.from_dict(example()), details=False)
        interval = result["performance"]["decode"]["device_inter_token_interval"]
        self.assertGreater(interval["last_seconds"], interval["first_seconds"])

    def test_output_length_one_has_no_decode(self):
        data = example()
        data["serving"]["output_length"]["value"] = 1
        decode = estimate(Config.from_dict(data), False)["performance"]["decode"]
        self.assertEqual(decode["steps"], 0)
        self.assertIsNone(decode["device_inter_token_interval"]["mean_seconds"])

    def test_capacity_overflow_keeps_theoretical_performance(self):
        data = example()
        data["hardware"]["device_memory"]["capacity_bytes"] = 1
        result = estimate(Config.from_dict(data), False)
        self.assertFalse(result["capacity"]["capacity_feasible"])
        self.assertTrue(result["capacity"]["performance_is_theoretical"])
        self.assertIsNotNone(result["performance"])

    def test_unmodeled_parameters_are_capacity_only(self):
        data = example()
        data["model"]["extra"] = {"parameter_count": 1_000_000, "sharding": "tp_ep"}
        result = estimate(Config.from_dict(data), False)
        self.assertFalse(result["validity"]["performance_complete"])
        self.assertEqual(result["model"]["parameters"], 5_803_037_248)
        self.assertGreater(result["performance"]["known_latency_lower_bound_seconds"], 0)

    def test_unknown_operator_and_wrong_slot_are_rejected(self):
        unknown = example()
        unknown["model"]["layer_pattern"][0]["attention"]["type"] = "mystery"
        with self.assertRaisesRegex(ValueError, "unknown operator"):
            Config.from_dict(unknown)
        wrong = example()
        wrong["model"]["layer_pattern"][0]["attention"] = {"type": "gated_ffn"}
        with self.assertRaisesRegex(ValueError, "not attention slot"):
            Config.from_dict(wrong)


if __name__ == "__main__":
    unittest.main()
