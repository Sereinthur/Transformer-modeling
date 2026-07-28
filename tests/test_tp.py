import copy
import unittest

from transformer_modeling import Config, estimate
from transformer_modeling.communication import (
    bus_all_reduce_seconds, crossbar_all_reduce_seconds,
    mesh_all_reduce_seconds, ring_all_reduce_seconds,
)

from helpers import example, parallel, phase_operators


class TensorParallelTests(unittest.TestCase):
    def test_ring_reference(self):
        self.assertAlmostEqual(ring_all_reduce_seconds(16 * 2**20, 8, 100e9, 2e-6), 321.60128e-6)

    def test_other_topology_formulas_are_distinct(self):
        args = (16 * 2**20, 8, 100e9, 2e-6)
        values = {round(function(*args), 12) for function in (ring_all_reduce_seconds, bus_all_reduce_seconds, crossbar_all_reduce_seconds, mesh_all_reduce_seconds)}
        self.assertGreater(len(values), 2)

    def test_tp_rewrites_standard_attention_shapes(self):
        for tp in (2, 4, 8):
            with self.subTest(tp=tp):
                result = estimate(Config.from_dict(parallel(example(), tp=tp)), True)
                attention = next(item for item in phase_operators(result["performance"]["prefill"]) if item["type"] == "standard_attention")
                qkv = next(item for item in attention["suboperators"] if item["name"] == "layers.qkv_projection")
                # gemm_shape按[M, N, K]输出。GQA下Q宽度与KV宽度分别按TP规则改写。
                self.assertEqual(qkv["gemm_shape"][1], (4096 + 2 * 1024) // tp)
                self.assertEqual(qkv["gemm_shape"][2], 4096)

    def test_tp_inserts_attention_ffn_and_logits_collectives(self):
        result = estimate(Config.from_dict(parallel(example(), tp=4)), True)
        operators = phase_operators(result["performance"]["prefill"])
        collectives = [item for operator in operators for item in operator["communication"]]
        kinds = [item["collective"]["type"] for item in collectives]
        self.assertGreaterEqual(kinds.count("all_reduce"), 3)
        self.assertEqual(kinds.count("all_gather"), 1)

        embedding = next(item for item in operators if item["type"] == "token_embedding")
        self.assertEqual(
            embedding["communication"][0]["name"],
            "embedding_hidden_all_reduce",
        )
        self.assertEqual(
            embedding["communication"][0]["collective"]["api_payload_bytes_per_rank_per_occurrence"],
            2 * 1024 * 4096 * 2,
        )

    def test_all_topologies_flow_through_operator_communication(self):
        for topology in ("ring", "bus", "crossbar", "mesh"):
            with self.subTest(topology=topology):
                data = parallel(example(), tp=4)
                data["hardware"]["interconnect"]["topology"] = topology
                result = estimate(Config.from_dict(data), True)
                attention = next(item for item in phase_operators(result["performance"]["prefill"]) if item["type"] == "standard_attention")
                self.assertEqual(attention["communication"][0]["collective"]["topology"], topology)

    def test_invalid_heads_and_vocab_are_rejected(self):
        heads = parallel(example(), tp=4)
        heads["model"]["layer_pattern"][0]["attention"]["query_heads"] = 10
        with self.assertRaisesRegex(ValueError, "query_heads"):
            Config.from_dict(heads)
        vocab = parallel(example(), tp=4)
        vocab["model"]["dimensions"]["padded_vocab_size"] += 1
        with self.assertRaisesRegex(ValueError, "padded_vocab_size"):
            Config.from_dict(vocab)

    def test_missing_interconnect_is_rejected(self):
        data = parallel(example(), tp=2)
        data["hardware"]["interconnect"]["effective_channel_bandwidth_bytes_per_second"] = None
        with self.assertRaisesRegex(ValueError, "bandwidth"):
            Config.from_dict(data)


if __name__ == "__main__":
    unittest.main()
