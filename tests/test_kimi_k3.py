import copy
import unittest

from transformer_modeling import Config, estimate
from transformer_modeling.models import get_preset

from helpers import example, parallel, phase_operators


def k3_data(tp=1, ep=1, pp=1):
    data = example()
    data["model"] = get_preset("kimi-k3-official")["model"]
    data["hardware"]["compute"]["throughput"] = {
        "bf16_dense_ops_per_second": 1e15,
        "mxfp4_mxfp8_ops_per_second": 2e15,
    }
    return parallel(data, tp=tp, ep=ep, pp=pp)


def op(layer, type_id):
    return next(item.operator for item in layer.operations if item.operator.type == type_id)


class KimiK3Tests(unittest.TestCase):
    def test_removed_draft_id_has_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "use kimi-k3-official"):
            get_preset("kimi-k3-draft")

    def test_exact_v3_layer_order_and_operator_counts(self):
        config = Config.from_dict(k3_data())
        layers = config.model.expanded_layers()
        self.assertEqual(len(layers), 93)
        self.assertEqual([i + 1 for i, layer in enumerate(layers) if any(item.operator.type == "gated_mla" for item in layer.operations)], [*range(4, 93, 4), 93])
        self.assertEqual(op(layers[0], "gated_ffn").implementation, "situ_glu")
        self.assertTrue(all(op(layer, "moe").type == "moe" for layer in layers[1:]))
        result = estimate(config, False)
        mix = result["model"]["operator_mix"]
        self.assertEqual({key: mix[key] for key in ("kda", "gated_mla", "gated_ffn", "moe", "attnres")}, {"kda": 69, "gated_mla": 24, "gated_ffn": 1, "moe": 92, "attnres": 93})
        self.assertEqual(result["model"]["special_operator_estimation"]["attnres"]["formula_confidence"], "approximate")

    def test_attnres_is_an_ordered_operator_and_drives_pp_payload(self):
        data = k3_data(pp=2)
        data["parallelism"]["pipeline_stage_boundaries"] = [48]
        data["serving"]["batch_size"] = 1
        data["serving"]["prompt_length"] = {"distribution": "fixed", "value": 64}
        config = Config.from_dict(data)
        self.assertEqual(op(config.model.expanded_layers()[0], "attnres").get("block_size"), 12)
        result = estimate(config, True)
        ordered = phase_operators(result["performance"]["prefill"])
        self.assertEqual(ordered[1]["type"], "attnres")  # embedding then the first layer's AttnRes
        self.assertEqual(sum(item["occurrences"] for item in ordered if item["type"] == "attnres"), 186)
        stage = result["capacity"]["per_stage"][0]
        self.assertEqual(stage["pipeline_state_width_elements"], config.model.state_width_after_layer(47))
        self.assertGreater(stage["pipeline_state_payload_bytes"], 64 * config.model.hidden_size * config.model.activation_bytes)

    def test_capacity_and_state_regression(self):
        base = estimate(Config.from_dict(k3_data()), False)
        self.assertGreater(base["capacity"]["required_bytes_per_critical_rank"], 1.3 * 2**40)
        self.assertEqual(base["model"]["metadata"]["published_parameter_reference"], 2_780_000_000_000)
        long_data = k3_data()
        long_data["serving"]["output_length"]["value"] = 512
        long = estimate(Config.from_dict(long_data), False)
        first = base["capacity"]["per_stage"][0]["states_by_operator"]
        later = long["capacity"]["per_stage"][0]["states_by_operator"]
        self.assertEqual(first["kda_state_bytes"], later["kda_state_bytes"])
        self.assertGreater(later["mla_latent_kv_cache_bytes"], first["mla_latent_kv_cache_bytes"])

    def test_v3_has_no_global_hidden_flow(self):
        legacy = copy.deepcopy(k3_data())
        legacy["model"]["hidden_state_flow"] = {"type": "attnres", "block_size": 12}
        with self.assertRaisesRegex(ValueError, "configuration version expired"):
            Config.from_dict(legacy)


if __name__ == "__main__":
    unittest.main()
