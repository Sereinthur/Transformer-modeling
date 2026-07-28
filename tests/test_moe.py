import copy
import unittest

from transformer_modeling import Config, estimate

from helpers import example, parallel, phase_operators


class MoeTests(unittest.TestCase):
    def setUp(self):
        self.data = example("moe_qwen3_30b_a3b")

    def test_moe_reports_router_experts_and_combine(self):
        result = estimate(Config.from_dict(self.data), True)
        moe = next(item for item in phase_operators(result["performance"]["prefill"]) if item["type"] == "moe")
        names = {item["name"] for item in moe["suboperators"]}
        self.assertIn("layers.moe_router", names)
        self.assertIn("layers.moe_expert_down", names)
        self.assertIn("layers.moe_combine", names)

    def test_ep_shards_expert_capacity_and_inserts_two_all_to_all(self):
        base = estimate(Config.from_dict(self.data), True)
        data = parallel(self.data, ep=4)
        ep = estimate(Config.from_dict(data), True)
        base_weights = base["capacity"]["per_stage"][0]["weights_bytes"]
        ep_weights = ep["capacity"]["per_stage"][0]["weights_bytes"]
        self.assertLess(ep_weights, base_weights)
        moe = next(item for item in phase_operators(ep["performance"]["prefill"]) if item["type"] == "moe")
        kinds = [item["collective"]["type"] for item in moe["communication"]]
        self.assertEqual(kinds.count("all_to_all"), 2)

    def test_ep_smaller_than_batch_uses_all_ranks(self):
        data = parallel(self.data, ep=4)
        data["serving"]["batch_size"] = 8
        phase = estimate(Config.from_dict(data), False)["performance"]["prefill"]
        self.assertEqual(phase["active_ep_ranks"], 4)
        self.assertEqual(phase["ep_utilization"], 1)

    def test_small_batch_reports_idle_ep_ranks(self):
        data = parallel(self.data, ep=8)
        data["serving"]["batch_size"] = 2
        phase = estimate(Config.from_dict(data), False)["performance"]["prefill"]
        self.assertEqual(phase["active_ep_ranks"], 2)
        self.assertEqual(phase["ep_utilization"], 0.25)

    def test_invalid_expert_and_tp_divisibility(self):
        data = parallel(self.data, ep=3)
        with self.assertRaisesRegex(ValueError, "expert_count"):
            Config.from_dict(data)
        data = parallel(self.data, tp=5)
        with self.assertRaises(ValueError):
            Config.from_dict(data)

    def test_ep_requires_moe_operator(self):
        with self.assertRaisesRegex(ValueError, "requires at least one moe"):
            Config.from_dict(parallel(example(), ep=2))


if __name__ == "__main__":
    unittest.main()
