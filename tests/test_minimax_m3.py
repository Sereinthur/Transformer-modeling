import copy
import unittest

from transformer_modeling import Config, estimate
from transformer_modeling.models import get_preset

from helpers import example, phase_operators


def m3_data():
    data = example()
    data["model"] = get_preset("minimax-m3")["model"]
    data["hardware"]["compute"]["throughput"] = {"bf16_dense_ops_per_second": 1e15}
    return data


class MiniMaxM3Tests(unittest.TestCase):
    def test_official_text_backbone_sequence_and_parameters(self):
        config = Config.from_dict(m3_data())
        layers = config.model.expanded_layers()
        self.assertEqual(len(layers), 60)
        self.assertEqual(
            [next(item.operator.type for item in layer.operations if item.operator.type.endswith("attention")) for layer in layers[:3]],
            ["standard_attention"] * 3,
        )
        self.assertEqual(
            [next(item.operator.type for item in layer.operations if item.operator.type.endswith("attention")) for layer in layers[3:]],
            ["minimax_sparse_attention"] * 57,
        )
        ffns = [next(item.operator.type for item in layer.operations if item.operator.type in {"gated_ffn", "moe"}) for layer in layers]
        self.assertEqual(ffns[:3], ["gated_ffn"] * 3)
        self.assertEqual(ffns[3:], ["moe"] * 57)
        result = estimate(config, False)
        self.assertGreater(result["model"]["parameters"], 425_000_000_000)
        self.assertLess(result["model"]["parameters"], 431_000_000_000)

    def test_msa_topk_reduces_main_attention_traffic_but_not_index_scan(self):
        baseline = estimate(Config.from_dict(m3_data()), True)
        reduced_data = copy.deepcopy(m3_data())
        reduced_data["model"]["layer_pattern"][0]["operations"][1]["operator"]["topk_blocks"] = 1
        reduced = estimate(Config.from_dict(reduced_data), True)

        def work(result, name):
            operator = next(item for item in phase_operators(result["performance"]["decode"]["first_step"])
                            if item["type"] == "minimax_sparse_attention")
            return next(item for item in operator["suboperators"] if item["name"] == name)

        self.assertLess(
            work(reduced, "layers.msa_sparse_attention")["hbm_payload_bytes"],
            work(baseline, "layers.msa_sparse_attention")["hbm_payload_bytes"],
        )
        self.assertEqual(
            work(reduced, "layers.msa_indexer_score")["hbm_payload_bytes"],
            work(baseline, "layers.msa_indexer_score")["hbm_payload_bytes"],
        )

    def test_msa_validates_rope_dimension(self):
        data = m3_data()
        data["model"]["layer_pattern"][0]["operations"][1]["operator"]["rope_dim"] = 129
        with self.assertRaisesRegex(ValueError, "rope_dim"):
            Config.from_dict(data)


if __name__ == "__main__":
    unittest.main()
