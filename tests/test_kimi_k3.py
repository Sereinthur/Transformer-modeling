import copy
import unittest

from transformer_modeling import Config, estimate
from transformer_modeling.models import get_preset

from helpers import example, parallel, phase_operators


def k3_data(scenario="base", tp=1, ep=1, pp=1):
    data = example("kimi_k3_base_tp64")
    data["model"] = get_preset("kimi-k3-draft", scenario)["model"]
    data["hardware"]["compute"]["throughput"] = {"mxfp4_mxfp8_ops_per_second": 1e15}
    return parallel(data, tp=tp, ep=ep, pp=pp)


class KimiK3Tests(unittest.TestCase):
    def test_all_scenarios_reconcile_to_2_8t(self):
        for scenario in ("compact", "base", "deep_wide"):
            with self.subTest(scenario=scenario):
                result = estimate(Config.from_dict(k3_data(scenario)), False)
                self.assertEqual(result["model"]["parameters"], 2_800_000_000_000)
                self.assertFalse(result["validity"]["performance_complete"])

    def test_k3_tp1_is_over_one_tib_not_twelve_gib(self):
        result = estimate(Config.from_dict(k3_data()), False)
        required = result["capacity"]["required_bytes_per_critical_rank"]
        self.assertGreater(required, 1.3 * 2**40)
        self.assertFalse(result["capacity"]["capacity_feasible"])

    def test_pattern_is_three_kda_one_mla(self):
        result = estimate(Config.from_dict(k3_data()), False)
        mix = result["model"]["operator_mix"]
        self.assertEqual(mix["kda"], 48)
        self.assertEqual(mix["gated_mla"], 16)
        self.assertEqual(mix["moe"], 64)
        self.assertEqual(mix["attnres"], 129)

    def test_attnres_is_pre_sublayer_temporary_working_set(self):
        result = estimate(Config.from_dict(k3_data(tp=64)), True)
        operators = phase_operators(result["performance"]["prefill"])
        first_attnres = next(item for item in operators if item["type"] == "attnres")
        first_norm = next(item for item in operators if item["type"] == "rms_norm")
        self.assertLess(operators.index(first_attnres), operators.index(first_norm))

        attnres = [item for item in operators if item["type"] == "attnres"]
        self.assertEqual(sum(item["occurrences"] for item in attnres), 129)
        self.assertEqual(sum(item["capacity"]["local_parameters"] for item in attnres), 1_849_344)
        self.assertTrue(all(item["capacity"]["persistent_state_bytes"] == 0 for item in attnres))
        self.assertTrue(all(item["capacity"]["temporary_bytes"] > 0 for item in attnres))

    def test_capacity_uses_same_operator_workspace_peak(self):
        result = estimate(Config.from_dict(k3_data(tp=64)), True)
        stage = result["capacity"]["per_stage"][0]
        operators = phase_operators(result["performance"]["prefill"])
        expected = max(
            item["capacity"]["temporary_bytes"]
            + max((comm["collective"]["api_payload_bytes_per_rank_per_occurrence"]
                   for comm in item["communication"]), default=0)
            for item in operators
        )
        self.assertEqual(stage["combined_workspace_peak_bytes"], expected)

    def test_kda_state_constant_and_mla_cache_grows(self):
        short = estimate(Config.from_dict(k3_data()), False)
        data = k3_data()
        data["serving"]["output_length"]["value"] = 512
        long = estimate(Config.from_dict(data), False)
        s1 = short["capacity"]["per_stage"][0]["states_by_operator"]
        s2 = long["capacity"]["per_stage"][0]["states_by_operator"]
        self.assertEqual(s1["kda_state_bytes"], s2["kda_state_bytes"])
        self.assertGreater(s2["mla_latent_kv_cache_bytes"], s1["mla_latent_kv_cache_bytes"])

    def test_prefix_hits_reduce_known_prefill_latency(self):
        data = k3_data()
        data["serving"]["prompt_length"]["value"] = 1024
        base = estimate(Config.from_dict(data), False)["performance"]["prefill"]["latency_seconds"]
        hit = copy.deepcopy(data)
        hit["serving"]["prefix_cache"].update(
            kda_state_hit_rate=1, kda_cached_prefix_tokens=768,
            mla_prefix_hit_rate=1, mla_average_matched_tokens=768,
        )
        faster = estimate(Config.from_dict(hit), False)["performance"]["prefill"]["latency_seconds"]
        self.assertLess(faster, base)

    def test_k3_can_use_pp_and_ep_without_model_name_branch(self):
        data = k3_data(tp=2, ep=8, pp=2)
        result = estimate(Config.from_dict(data), True)
        self.assertEqual(len(result["performance"]["prefill"]["stages"]), 2)
        moe = next(item for item in phase_operators(result["performance"]["prefill"]) if item["type"] == "moe")
        self.assertEqual(sum(item["collective"]["type"] == "all_to_all" for item in moe["communication"]), 2)

    def test_missing_native_mx_throughput_keeps_capacity_only(self):
        data = k3_data()
        data["hardware"]["compute"]["throughput"] = {"int4_dense_ops_per_second": 1e15}
        result = estimate(Config.from_dict(data), False)
        self.assertIsNone(result["performance"])
        self.assertGreater(result["capacity"]["required_bytes_per_critical_rank"], 0)


if __name__ == "__main__":
    unittest.main()
