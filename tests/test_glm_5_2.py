import copy
import unittest

from transformer_modeling import Config, estimate
from transformer_modeling.models import get_preset

from helpers import example, parallel


def glm_data(tp=8, ep=8):
    data = example()
    data["model"] = get_preset("glm-5.2")["model"]
    data["hardware"]["compute"]["throughput"] = {"bf16_dense_ops_per_second": 1e15}
    return parallel(data, tp=tp, ep=ep, pp=1)


class GLM52Tests(unittest.TestCase):
    def test_official_text_backbone_sequence(self):
        model = Config.from_dict(glm_data()).model
        layers = model.expanded_layers()
        self.assertEqual(len(layers), 78)
        attention = [next(item.operator for item in layer.operations if item.operator.type == "dsa_attention") for layer in layers]
        ffns = [next(item.operator.type for item in layer.operations if item.operator.type in {"gated_ffn", "moe"}) for layer in layers]
        self.assertEqual([item.get("indexer_mode") for item in attention[:7]], ["full", "full", "full", "shared", "shared", "shared", "full"])
        self.assertEqual(sum(item.get("indexer_mode") == "full" for item in attention), 21)
        self.assertEqual(sum(item.get("indexer_mode") == "shared" for item in attention), 57)
        self.assertEqual(ffns[:3], ["gated_ffn"] * 3)
        self.assertEqual(ffns[3:], ["moe"] * 75)

    def test_parameter_anchor_and_indexshare_mode_validation(self):
        result = estimate(Config.from_dict(glm_data()), False)
        self.assertGreater(result["model"]["parameters"], 744_000_000_000 * .995)
        self.assertLess(result["model"]["parameters"], 744_000_000_000 * 1.01)
        self.assertEqual(result["model"]["operator_mix"]["dsa_attention"], 78)
        invalid = copy.deepcopy(glm_data())
        invalid["model"]["layer_prefix"][0]["operations"][1]["operator"]["indexer_mode"] = "invalid"
        with self.assertRaisesRegex(ValueError, "indexer_mode must be full or shared"):
            Config.from_dict(invalid)

    def test_dsa_mode_is_an_independent_editable_operator_parameter(self):
        baseline = estimate(Config.from_dict(glm_data()), False)
        edited = copy.deepcopy(glm_data())
        # The second item is the DSA operator; changing just this one layer's
        # IndexShare mode must affect the backend estimate.
        edited["model"]["layer_prefix"][0]["operations"][1]["operator"]["indexer_mode"] = "shared"
        changed = estimate(Config.from_dict(edited), False)
        self.assertLess(changed["model"]["parameters"], baseline["model"]["parameters"])
        self.assertLess(
            changed["performance"]["decode"]["first_step"]["ops"]["executed"],
            baseline["performance"]["decode"]["first_step"]["ops"]["executed"],
        )

    def test_dsa_topk_reduces_sparse_attention_cache_reads(self):
        baseline = estimate(Config.from_dict(glm_data()), True)
        reduced = copy.deepcopy(glm_data())
        for layer in (
            *reduced["model"].get("layer_prefix", []),
            *reduced["model"]["layer_pattern"],
            *reduced["model"].get("layer_suffix", []),
        ):
            for item in layer["operations"]:
                if item["operator"]["type"] == "dsa_attention":
                    item["operator"]["index_topk"] = 1
        changed = estimate(Config.from_dict(reduced), True)

        def sparse_hbm(result):
            stages = result["performance"]["decode"]["first_step"]["stages"]
            return sum(
                item["hbm_payload_bytes"]
                for stage in stages for operator in stage["operators"]
                if operator["type"] == "dsa_attention"
                for item in operator["suboperators"]
                if item["name"] == "layers.dsa_sparse_attention"
            )

        self.assertLess(sparse_hbm(changed), sparse_hbm(baseline))

    def test_embedding_table_is_independently_parameterized(self):
        baseline = estimate(Config.from_dict(glm_data()), False)
        edited = copy.deepcopy(glm_data())
        edited["model"]["embedding"].update({
            "vocab_size": 154_888,
            "embedding_dim": 6144,
            "weight_dtype": "fp8",
        })
        changed = estimate(Config.from_dict(edited), False)
        # Eight extra rows at width 6144, plus the input table is now FP8.
        self.assertEqual(
            changed["model"]["parameters"] - baseline["model"]["parameters"],
            8 * 6144,
        )
        invalid_width = copy.deepcopy(edited)
        invalid_width["model"]["embedding"]["embedding_dim"] = 4096
        with self.assertRaisesRegex(ValueError, "embedding_dim must equal"):
            Config.from_dict(invalid_width)
        tied = copy.deepcopy(edited)
        tied["model"]["embedding"]["tied_lm_head"] = True
        with self.assertRaisesRegex(ValueError, "a tied token_embedding"):
            Config.from_dict(tied)


if __name__ == "__main__":
    unittest.main()
