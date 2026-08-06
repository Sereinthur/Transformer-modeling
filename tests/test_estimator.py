import copy
import unittest

from transformer_modeling import Config, estimate
from transformer_modeling.models import get_preset

from helpers import example, phase_operators


class EstimatorTests(unittest.TestCase):
    def test_schema_v3_is_required(self):
        data = example()
        data["schema_version"] = 1
        with self.assertRaisesRegex(ValueError, "configuration version expired"):
            Config.from_dict(data)

    def test_fixed_skeleton_expands_cyclic_pattern(self):
        data = example()
        data["model"] = get_preset("kimi-k3-official")["model"]
        data["hardware"]["compute"]["throughput"] = {"bf16_dense_ops_per_second": 1e15}
        config = Config.from_dict(data)
        types = [next(op.operator.type for op in layer.operations if op.operator.type in {"kda", "gated_mla"}) for layer in config.model.expanded_layers()]
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

    def test_single_request_decode_throughput_uses_step_latency(self):
        from helpers import parallel

        result = estimate(Config.from_dict(parallel(example(), pp=2)), False)
        decode = result["performance"]["decode"]
        interval = decode["device_inter_token_interval"]
        self.assertEqual(interval["steady_state_seconds"], interval["mean_seconds"])
        self.assertGreaterEqual(
            interval["pipeline_service_interval_seconds"], interval["steady_state_seconds"]
        )

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
        unknown["model"]["layer_pattern"][0]["operations"][1]["operator"]["type"] = "mystery"
        with self.assertRaisesRegex(ValueError, "unknown operator"):
            Config.from_dict(unknown)
        reordered = example()
        reordered["model"]["layer_pattern"][0]["operations"][1]["operator"] = {"type": "gated_ffn", "intermediate_size": 11008}
        # v3 intentionally permits FFN/MoE at any main-backbone position.
        self.assertEqual(Config.from_dict(reordered).model.expanded_layers()[0].main_operators[1].type, "gated_ffn")

    def test_ordered_layer_accepts_moe_and_rejects_legacy_edges(self):
        data = example()
        data["model"]["layer_pattern"] = [{
            "repeat": 1,
            "operations": [
                {"id": "pre", "operator": {"type": "rms_norm"}},
                {"id": "moe_before_attention", "operator": {
                    "type": "moe", "expert_count": 8,
                    "experts_per_token": 2, "expert_intermediate_size": 128,
                }},
                {"id": "attention", "operator": data["model"]["layer_pattern"][0]["operations"][1]["operator"]},
                {"id": "ffn", "operator": data["model"]["layer_pattern"][0]["operations"][4]["operator"]},
            ],
        }]
        config = Config.from_dict(data)
        self.assertEqual(
            [operator.type for operator in config.model.expanded_layers()[0].main_operators],
            ["rms_norm", "moe", "standard_attention", "gated_ffn"],
        )
        result = estimate(config, details=True)
        self.assertEqual(result["model"]["operator_mix"]["moe"], config.model.layer_count)

        bad = copy.deepcopy(data)
        bad["model"]["layer_pattern"][0]["residual_connections"] = []
        with self.assertRaisesRegex(ValueError, "configuration version expired"):
            Config.from_dict(bad)


if __name__ == "__main__":
    unittest.main()
