import json
import unittest
from copy import deepcopy
from pathlib import Path

from transformer_modeling import Config, estimate


ROOT = Path(__file__).resolve().parents[1]


def example():
    return json.loads((ROOT / "examples" / "single_chip_gqa.json").read_text(encoding="utf-8"))


def first_operator(result, type_id):
    for operator in result["performance"]["prefill"]["stages"][0]["operators"]:
        if operator["type"] == type_id:
            return operator
    raise AssertionError(f"missing {type_id}")


class ParameterContractTests(unittest.TestCase):
    def test_operator_kv_cache_dtype_overrides_model_default(self):
        base = example()
        attention = base["model"]["layer_pattern"][0]["operations"][1]["operator"]
        attention["kv_cache_dtype"] = "fp8"
        override = estimate(Config.from_dict(base), details=True)
        override_cache = first_operator(override, "standard_attention")["capacity"]["state_breakdown"]["kv_cache_bytes"]

        inherited = deepcopy(base)
        inherited["model"]["layer_pattern"][0]["operations"][1]["operator"].pop("kv_cache_dtype")
        inherited["model"]["dtype"]["kv_cache"] = "bf16"
        model_default = estimate(Config.from_dict(inherited), details=True)
        default_cache = first_operator(model_default, "standard_attention")["capacity"]["state_breakdown"]["kv_cache_bytes"]
        self.assertLess(override_cache, default_cache)

    def test_operator_weight_dtype_selects_item_throughput(self):
        data = example()
        data["hardware"]["compute"]["throughput"]["fp8_dense_ops_per_second"] = 8e15
        attention = data["model"]["layer_pattern"][0]["operations"][1]["operator"]
        attention["weight_dtype"] = "fp8"
        result = estimate(Config.from_dict(data), details=True)
        work = first_operator(result, "standard_attention")["suboperators"]
        gemms = [item for item in work if item["kind"] == "gemm"]
        self.assertTrue(gemms)
        self.assertTrue(all(item["compute_dtype"] == "fp8" for item in gemms))
        self.assertTrue(all(item["compute_throughput_source"] == "peak" for item in gemms))

    def test_norm_dtype_is_validated_and_bucketed(self):
        data = example()
        norm = data["model"]["layer_pattern"][0]["operations"][0]["operator"]
        norm["weight_dtype"] = "fp8"
        result = estimate(Config.from_dict(data), details=True)
        norm_result = first_operator(result, "rms_norm")
        self.assertEqual(norm_result["capacity"]["local_parameters"], 4096)


if __name__ == "__main__":
    unittest.main()
