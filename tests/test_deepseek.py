import copy
import unittest

from transformer_modeling import Config, estimate
from transformer_modeling.models import get_preset

from helpers import example, parallel, phase_operators


def v4_data(variant="pro", tp=8, ep=8, pp=1):
    data = example("kimi_k3_base_tp64")
    data["model"] = get_preset(f"deepseek-v4-{variant}")["model"]
    data["hardware"]["compute"]["throughput"] = {"mxfp4_mxfp8_ops_per_second": 1e15}
    return parallel(data, tp=tp, ep=ep, pp=pp)


def attention_types(model):
    return [layer.attention.type for layer in model.expanded_layers()]


def operator(result, type_id, phase="prefill"):
    return next(
        item for item in phase_operators(result["performance"][phase])
        if item["type"] == type_id
    )


class DeepSeekV4Tests(unittest.TestCase):
    def test_presets_resolve_and_validate(self):
        for variant in ("pro", "flash"):
            with self.subTest(variant=variant):
                result = estimate(Config.from_dict(v4_data(variant)), True)
                self.assertGreater(result["capacity"]["required_bytes_per_critical_rank"], 0)
                csa = operator(result, "csa_attention")
                self.assertEqual(csa["confidence"], "low")
                self.assertTrue(csa["assumptions"])

    def test_layer_prefix_reproduces_documented_layer_order(self):
        pro = Config.from_dict(v4_data("pro")).model
        types = attention_types(pro)
        self.assertEqual(len(types), 61)
        # 前2层纯滑窗，第3层CSA，之后HCA/CSA交替，末层是CSA。
        self.assertEqual(types[:4], [
            "standard_attention", "standard_attention", "csa_attention", "hca_attention",
        ])
        self.assertEqual(types[60], "csa_attention")
        self.assertEqual(types.count("csa_attention"), 30)
        self.assertEqual(types.count("hca_attention"), 29)
        flash = Config.from_dict(v4_data("flash")).model
        flash_types = attention_types(flash)
        self.assertEqual(len(flash_types), 43)
        self.assertEqual(flash_types[42], "csa_attention")

    def test_prefix_layers_use_hash_routing_without_router_gemm(self):
        result = estimate(Config.from_dict(v4_data("pro")), True)
        moes = [item for item in phase_operators(result["performance"]["prefill"]) if item["type"] == "moe"]
        hashed = next(item for item in moes if item["occurrences"] == 3)
        learned = next(item for item in moes if item["occurrences"] == 58)
        self.assertNotIn("layers.moe_router", [item["name"] for item in hashed["suboperators"]])
        self.assertIn("layers.moe_router", [item["name"] for item in learned["suboperators"]])
        per_hashed = hashed["capacity"]["local_parameters"] / hashed["occurrences"]
        per_learned = learned["capacity"]["local_parameters"] / learned["occurrences"]
        self.assertLess(per_hashed, per_learned)

    def test_parameter_count_reconciles_to_target(self):
        for variant, target in (("pro", 1_600_000_000_000), ("flash", 284_000_000_000)):
            with self.subTest(variant=variant):
                result = estimate(Config.from_dict(v4_data(variant)), False)
                parameters = result["model"]["parameters"]
                self.assertGreaterEqual(parameters, target)
                self.assertLess(parameters, target * 1.01)

    def test_compressed_attention_is_cheaper_than_global_attention(self):
        data = v4_data("pro")
        # 前置层的滑窗放开成全局注意力，作为同维度全局Attention的对照。
        for prefix in data["model"]["layer_prefix"]:
            prefix["attention"]["sliding_window"] = 0
        result = estimate(Config.from_dict(data), True)
        latency = {
            type_id: operator(result, type_id)["time_seconds"]["estimated"]
            / operator(result, type_id)["occurrences"]
            for type_id in ("standard_attention", "csa_attention", "hca_attention")
        }
        self.assertLess(latency["hca_attention"], latency["csa_attention"])
        self.assertLess(latency["csa_attention"], latency["standard_attention"])

    def test_compressed_kv_cache_is_much_smaller_than_full_mqa_cache(self):
        compressed = estimate(Config.from_dict(v4_data("pro")), False)
        states = compressed["capacity"]["per_stage"][0]["states_by_operator"]
        cached = sum(
            states[key] for key in
            ("compressed_kv_cache_bytes", "sliding_window_kv_cache_bytes", "indexer_key_cache_bytes")
        )
        # 用同样的头数、head_dim把压缩层换成全量MQA注意力做对照。
        dense = v4_data("pro")
        full = {
            "type": "standard_attention", "implementation": "flash_attention",
            "query_heads": 128, "kv_heads": 1, "head_dim": 512, "qk_rope_head_dim": 64,
            "q_lora_rank": 1536, "o_lora_rank": 1536, "weight_dtype": "mxfp8",
            "query_width_equals_hidden": False,
        }
        for group in (*dense["model"]["layer_prefix"], *dense["model"]["layer_pattern"]):
            group["attention"] = copy.deepcopy(full)
        baseline = estimate(Config.from_dict(dense), False)
        reference = baseline["capacity"]["per_stage"][0]["states_by_operator"]["kv_cache_bytes"]
        self.assertLess(cached * 3, reference)

    def test_mhc_temporary_bytes_scale_with_channels(self):
        def temporary(channels):
            data = v4_data("pro")
            for group in (*data["model"]["layer_prefix"], *data["model"]["layer_pattern"]):
                group["residual"]["channels"] = channels
            result = estimate(Config.from_dict(data), True)
            item = operator(result, "mhc")
            return item["capacity"]["temporary_bytes"]

        self.assertAlmostEqual(temporary(8) / temporary(4), 2.0, places=6)

    def test_routed_expert_dtype_drives_weight_bytes(self):
        def weights(dtype):
            data = v4_data("pro")
            for group in (*data["model"]["layer_prefix"], *data["model"]["layer_pattern"]):
                group["ffn"]["routed_expert_weight_dtype"] = dtype
            result = estimate(Config.from_dict(data), False)
            return result["capacity"]["per_stage"][0]

        fp4 = weights("mxfp4")
        fp8 = weights("mxfp8")
        self.assertEqual(set(fp4["weights_by_dtype"]), {"mxfp4", "mxfp8"})
        ratio = fp8["weights_bytes"] / fp4["weights_bytes"]
        self.assertGreater(ratio, 1.6)
        self.assertLess(ratio, 2.1)


if __name__ == "__main__":
    unittest.main()
