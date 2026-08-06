import copy
import unittest

from transformer_modeling import Config, estimate
from transformer_modeling.parallel.pipeline import layer_partition, pipeline_schedule

from helpers import example, parallel


class PipelineParallelTests(unittest.TestCase):
    def test_basic_schedule_formula(self):
        schedule = pipeline_schedule([1, 1], [0, 0], 4)
        self.assertEqual(schedule["makespan_seconds"], 5)
        self.assertEqual(layer_partition(7, 3), [3, 2, 2])

    def test_schedule_serializes_shared_stage_link(self):
        schedule = pipeline_schedule([1, 1], [10, 0], 2)
        self.assertEqual(schedule["makespan_seconds"], 22)
        self.assertEqual(schedule["completion_matrix_seconds"][1], [12, 22])

    def test_pp_produces_stages_and_critical_capacity(self):
        result = estimate(Config.from_dict(parallel(example(), pp=4)), True)
        self.assertEqual(len(result["performance"]["prefill"]["stages"]), 4)
        self.assertEqual(len(result["capacity"]["per_stage"]), 4)
        self.assertIn(result["capacity"]["critical_stage_index"], range(4))

    def test_tp_ep_pp_device_product(self):
        data = parallel(example("moe_qwen3_30b_a3b"), tp=2, ep=4, pp=2)
        config = Config.from_dict(data)
        self.assertEqual(config.hardware.device_count, 16)
        self.assertEqual(estimate(config, False)["parallelism"]["device_count"], 16)

    def test_microbatches_reduce_balanced_pipeline_bubble(self):
        one = estimate(Config.from_dict(parallel(example(), pp=2, microbatches=1)), False)
        data = parallel(example(), pp=2, microbatches=4)
        data["serving"]["batch_size"] = 4
        four = estimate(Config.from_dict(data), False)
        self.assertLess(four["performance"]["prefill"]["pipeline_schedule"]["bubble_fraction"], one["performance"]["prefill"]["pipeline_schedule"]["bubble_fraction"])

    def test_slow_pipeline_link_increases_latency(self):
        data = parallel(example(), pp=2)
        fast = estimate(Config.from_dict(data), False)["performance"]["prefill"]["latency_seconds"]
        slow = copy.deepcopy(data)
        slow["hardware"]["interconnect"]["pipeline_effective_bandwidth_bytes_per_second"] = 1e6
        slower = estimate(Config.from_dict(slow), False)["performance"]["prefill"]["latency_seconds"]
        self.assertGreater(slower, fast)

    def test_manual_stage_boundaries_override_balancer(self):
        data = parallel(example(), pp=3)
        data["parallelism"]["pipeline_stage_boundaries"] = [4, 20]
        result = estimate(Config.from_dict(data), True)
        self.assertEqual([
            sum(item["occurrences"] for item in stage["operators"] if item["type"] == "rms_norm") // 2
            for stage in result["performance"]["prefill"]["stages"]
        ], [4, 16, 12])

        bad = copy.deepcopy(data)
        bad["parallelism"]["pipeline_stage_boundaries"] = [20, 4]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            Config.from_dict(bad)

    def test_invalid_pp_and_microbatch(self):
        with self.assertRaises(ValueError):
            Config.from_dict(parallel(example(), pp=100))
        data = parallel(example(), pp=2, microbatches=3)
        with self.assertRaisesRegex(ValueError, "divisible"):
            Config.from_dict(data)

    def test_invalid_pipeline_link_parameters_are_rejected(self):
        data = parallel(example(), pp=2)
        data["hardware"]["interconnect"]["pipeline_effective_bandwidth_bytes_per_second"] = 0
        with self.assertRaisesRegex(ValueError, "pipeline effective bandwidth"):
            Config.from_dict(data)
        data["hardware"]["interconnect"]["pipeline_effective_bandwidth_bytes_per_second"] = 1e9
        data["hardware"]["interconnect"]["pipeline_transfer_latency_seconds"] = -1
        with self.assertRaisesRegex(ValueError, "pipeline transfer latency"):
            Config.from_dict(data)

    def test_disaggregated_handoff_aggregates_all_pipeline_stages(self):
        data = parallel(example(), pp=2)
        data["deployment"] = {
            "mode": "disaggregated",
            "transfer": {"effective_bandwidth_bytes_per_second": 1e12},
        }
        result = estimate(Config.from_dict(data), False)
        pd = result["performance"]["pd_disaggregation"]
        expected = sum(stage["persistent_state_bytes"] for stage in result["capacity"]["per_stage"])
        self.assertEqual(pd["payload_bytes_per_rank"], expected)


if __name__ == "__main__":
    unittest.main()
