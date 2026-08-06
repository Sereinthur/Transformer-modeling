import copy
import unittest

from transformer_modeling import Config, estimate
from transformer_modeling.models import get_preset

from helpers import example, parallel, phase_operators


def v4_data(variant="pro", tp=8, ep=8, pp=1):
    data = example()
    data["model"] = get_preset(f"deepseek-v4-{variant}")["model"]
    data["hardware"]["compute"]["throughput"] = {"fp8_dense_ops_per_second": 1e15}
    return parallel(data, tp=tp, ep=ep, pp=pp)


def attention_types(model):
    attention = {"standard_attention", "sliding_window_attention", "kda", "gated_mla", "csa_attention", "hca_attention"}
    return [next(item.operator.type for item in layer.operations if item.operator.type in attention) for layer in model.expanded_layers()]


def mutable_ops(model, type_id):
    return [item["operator"] for segment in (*model.get("layer_prefix", []), *model.get("layer_pattern", []), *model.get("layer_suffix", [])) for item in segment["operations"] if item["operator"]["type"] == type_id]


class DeepSeekV4Tests(unittest.TestCase):
    def test_v3_layer_sequences(self):
        pro = Config.from_dict(v4_data("pro")).model
        self.assertEqual(attention_types(pro)[:4], ["hca_attention", "hca_attention", "csa_attention", "hca_attention"])
        self.assertEqual(attention_types(pro).count("hca_attention"), 31)
        self.assertEqual(attention_types(pro).count("csa_attention"), 30)
        self.assertNotIn("standard_attention", attention_types(pro))
        flash = Config.from_dict(v4_data("flash")).model
        self.assertEqual(len(attention_types(flash)), 43)
        self.assertEqual(attention_types(flash).count("sliding_window_attention"), 2)
        self.assertEqual(attention_types(flash).count("hca_attention"), 20)
        self.assertEqual(attention_types(flash).count("csa_attention"), 21)

    def test_special_mhc_is_independent_per_layer_operator(self):
        data = v4_data("pro", tp=1, ep=1, pp=2)
        data["parallelism"]["pipeline_stage_boundaries"] = [30]
        data["serving"]["batch_size"] = 1
        data["serving"]["prompt_length"] = {"distribution": "fixed", "value": 16}
        config = Config.from_dict(data)
        result = estimate(config, True)
        mhc = [item for item in phase_operators(result["performance"]["prefill"]) if item["type"] == "mhc"]
        self.assertEqual(len(mhc), 122)
        self.assertTrue(all(item["capacity"]["temporary_bytes"] > 0 for item in mhc))
        self.assertEqual(config.model.state_width_after_layer(29), 4 * config.model.hidden_size)
        stage = result["capacity"]["per_stage"][0]
        self.assertEqual(stage["pipeline_state_width_elements"], 4 * config.model.hidden_size)
        self.assertEqual(result["model"]["special_operator_estimation"]["mhc"]["occurrences"], 122)

    def test_mhc_replaces_residual_after_each_sublayer(self):
        model = Config.from_dict(v4_data("pro")).model
        first = [item.operator.type for item in model.expanded_layers()[0].operations]
        self.assertEqual(first, ["rms_norm", "hca_attention", "mhc", "rms_norm", "moe", "mhc"])
        self.assertNotIn("standard_residual", first)

    def test_mhc_parameter_changes_apply_to_cards(self):
        def temporary(channels):
            data = v4_data("pro")
            for operator in mutable_ops(data["model"], "mhc"):
                operator["channels"] = channels
            result = estimate(Config.from_dict(data), True)
            return sum(item["capacity"]["temporary_bytes"] for item in phase_operators(result["performance"]["prefill"]) if item["type"] == "mhc")
        self.assertAlmostEqual(temporary(8) / temporary(4), 2.0, places=6)

    def test_zero_sinkhorn_iterations_add_no_sinkhorn_work(self):
        data = v4_data("pro", tp=1, ep=1)
        for operator in mutable_ops(data["model"], "mhc"):
            operator["sinkhorn_iters"] = 0
        result = estimate(Config.from_dict(data), True)
        mhc = next(item for item in phase_operators(result["performance"]["prefill"])
                   if item["type"] == "mhc")
        default = estimate(Config.from_dict(v4_data("pro", tp=1, ep=1)), True)
        default_mhc = next(item for item in phase_operators(default["performance"]["prefill"])
                           if item["type"] == "mhc")
        self.assertLess(
            mhc["suboperators"][0]["ops"]["executed"],
            default_mhc["suboperators"][0]["ops"]["executed"],
        )

    def test_operator_replacement_and_validation(self):
        data = v4_data("flash")
        mutable_ops(data["model"], "sliding_window_attention")[0]["sliding_window"] = 0
        with self.assertRaisesRegex(ValueError, "sliding_window must be positive"):
            Config.from_dict(data)
        data = v4_data("pro")
        hca = mutable_ops(data["model"], "hca_attention")[0]
        hca.update(type="standard_attention", implementation="flash_attention", sliding_window=0, query_width_equals_hidden=False)
        result = estimate(Config.from_dict(data), False)
        self.assertEqual(result["model"]["operator_mix"]["standard_attention"], 2)

    def test_parameter_anchors_and_moe_dtypes(self):
        for variant, target in (("pro", 1_600_000_000_000), ("flash", 284_000_000_000)):
            result = estimate(Config.from_dict(v4_data(variant)), False)
            self.assertGreater(result["model"]["parameters"], target * .94)
            self.assertLess(result["model"]["parameters"], target * 1.01)
        data = v4_data("pro")
        for operator in mutable_ops(data["model"], "moe"):
            operator["routed_expert_weight_dtype"] = "fp8"
        fp8 = estimate(Config.from_dict(data), False)["capacity"]["per_stage"][0]["weights_bytes"]
        fp4 = estimate(Config.from_dict(v4_data("pro")), False)["capacity"]["per_stage"][0]["weights_bytes"]
        self.assertGreater(fp8, fp4)


if __name__ == "__main__":
    unittest.main()
