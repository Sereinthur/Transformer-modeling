import copy
import unittest

from transformer_modeling import Config, estimate

from helpers import example, parallel, phase_operators


def latent_data(tp=1, ep=1):
    return parallel(example("kimi_k3_base_tp64"), tp=tp, ep=ep)


def equivalent_width(data):
    """去掉latent_size并换成hidden域等效宽度：7168×1536=3584×3072。"""
    result = copy.deepcopy(data)
    for pattern in (
        *result["model"].get("layer_prefix", []),
        *result["model"]["layer_pattern"],
        *result["model"].get("layer_suffix", []),
    ):
        for item in pattern["operations"]:
            ffn = item["operator"]
            if ffn["type"] != "moe":
                continue
            ffn.pop("latent_size")
            ffn["expert_intermediate_size"] = 1536
            ffn["shared_expert_intermediate_size"] = 1536
    return result


class LatentMoETests(unittest.TestCase):
    def test_latent_projection_parameters_added(self):
        data = latent_data()
        baseline = equivalent_width(data)
        with_latent = estimate(Config.from_dict(data), False)["model"]["parameters"]
        without = estimate(Config.from_dict(baseline), False)["model"]["parameters"]
        self.assertEqual(with_latent - without, 92 * 2 * 7168 * 3584)

    def test_all_to_all_payload_uses_latent_width(self):
        data = latent_data(tp=2, ep=8)
        latent = estimate(Config.from_dict(data), True)
        widened = estimate(Config.from_dict(equivalent_width(data)), True)

        def payloads(result):
            moe = next(item for item in phase_operators(result["performance"]["prefill"])
                       if item["type"] == "moe")
            return [comm["collective"]["api_payload_bytes_per_rank_per_occurrence"]
                    for comm in moe["communication"]
                    if comm["collective"]["type"] == "all_to_all"]

        small, big = payloads(latent), payloads(widened)
        self.assertEqual(len(small), 2)
        for latent_payload, hidden_payload in zip(small, big):
            self.assertEqual(latent_payload * 2, hidden_payload)

    def test_latent_size_must_divide_tp(self):
        data = latent_data(tp=32)
        for pattern in (
            *data["model"].get("layer_prefix", []),
            *data["model"]["layer_pattern"],
            *data["model"].get("layer_suffix", []),
        ):
            for item in pattern["operations"]:
                if item["operator"]["type"] == "moe":
                    item["operator"]["latent_size"] = 3600
        with self.assertRaises(ValueError):
            estimate(Config.from_dict(data), False)


if __name__ == "__main__":
    unittest.main()
