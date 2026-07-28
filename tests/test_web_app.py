import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from web_app import VisualizationHandler


ROOT = Path(__file__).resolve().parents[1]


class VisualizationApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), VisualizationHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def get_json(self, route):
        with urlopen(f"{self.base_url}{route}", timeout=3) as response:
            return json.loads(response.read().decode("utf-8"))

    def post_json(self, route, payload, timeout=10):
        request = Request(
            f"{self.base_url}{route}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def test_main_page_and_generic_controls_are_served(self):
        with urlopen(f"{self.base_url}/", timeout=3) as response:
            content = response.read().decode("utf-8")
        for marker in (
            "Transformer 性能评估台", 'id="manual-model-preset"',
            'id="layer-pattern-editor"', 'id="add-pattern-row"',
            'id="tensor-parallel"', 'id="expert-parallel"',
            'id="pipeline-parallel"', 'id="calculate"',
        ):
            self.assertIn(marker, content)
        self.assertIn("Tensor / Expert / Pipeline Parallel", content)
        self.assertNotIn("本机实测", content)
        self.assertNotIn("local-mode-button", content)

    def test_removed_local_benchmark_endpoints_are_not_served(self):
        for route in ("/api/local-devices", "/api/calibration/jobs/missing"):
            with self.subTest(route=route), self.assertRaises(HTTPError) as context:
                urlopen(f"{self.base_url}{route}", timeout=3)
            self.assertEqual(context.exception.code, 404)
            context.exception.close()

    def test_frontend_es_modules_are_served(self):
        for route, marker in (
            ("/app.js", 'import { initialize } from "./js/main.js";'),
            ("/js/form.js", "export function buildConfig"),
            ("/js/results.js", "export function renderResult"),
            ("/js/manual-presets.js", "export async function initializeManualPresets"),
        ):
            with self.subTest(route=route):
                with urlopen(f"{self.base_url}{route}", timeout=3) as response:
                    content = response.read().decode("utf-8")
                self.assertIn(marker, content)

    def test_catalog_and_operator_endpoints(self):
        presets = self.get_json("/api/model-presets")["presets"]
        identifiers = {item["id"] for item in presets}
        self.assertIn("qwen3-0.6b", identifiers)
        self.assertIn("qwen3-30b-a3b", identifiers)
        self.assertIn("kimi-k3-draft", identifiers)
        catalog = self.get_json("/api/operator-catalog")
        operator_ids = {item["type"] for item in catalog["operators"]}
        self.assertTrue({"standard_attention", "kda", "gated_mla", "moe", "unmodeled"} <= operator_ids)

    def test_generic_model_definition_resolver(self):
        result = self.post_json(
            "/api/model-definitions/resolve",
            {"preset_id": "kimi-k3-draft", "scenario": "base"},
        )
        model = result["resolved_model"]
        self.assertEqual(model["dimensions"]["layer_count"], 64)
        self.assertEqual([item["attention"]["type"] for item in model["layer_pattern"]], ["kda", "gated_mla"])
        self.assertGreater(model["extra"]["parameter_count"], 0)

    def test_example_and_capacity_shortfall_still_return_performance(self):
        config = json.loads((ROOT / "examples" / "single_chip_gqa.json").read_text(encoding="utf-8"))
        config["hardware"]["device_memory"]["capacity_bytes"] = 1
        result = self.post_json("/api/estimate", config)
        self.assertEqual(result["schema_version"], 2)
        self.assertFalse(result["capacity"]["capacity_feasible"])
        self.assertTrue(result["validity"]["performance_is_theoretical"])
        self.assertIsNotNone(result["performance"])
        self.assertGreater(result["performance"]["first_token"]["ttft_seconds"], 0)

    def test_tp_ep_pp_and_all_topologies_through_api(self):
        base = json.loads((ROOT / "examples" / "moe_qwen3_30b_a3b.json").read_text(encoding="utf-8"))
        base["serving"]["output_length"]["value"] = 2
        base["parallelism"].update(tensor_parallel=2, expert_parallel=2, pipeline_parallel=2)
        base["hardware"]["device_count"] = 8
        for topology in ("ring", "bus", "crossbar", "mesh"):
            with self.subTest(topology=topology):
                config = json.loads(json.dumps(base))
                config["hardware"]["interconnect"]["topology"] = topology
                if topology == "mesh":
                    config["hardware"]["interconnect"].update(mesh_rows=2, mesh_columns=2)
                result = self.post_json("/api/estimate", config)
                self.assertEqual(result["parallelism"]["device_count"], 8)
                self.assertEqual(result["parallelism"]["topology"], topology)
                self.assertEqual(len(result["performance"]["prefill"]["stages"]), 2)
                communication = [
                    item["collective"]["type"]
                    for stage in result["performance"]["prefill"]["stages"]
                    for operator in stage["operators"]
                    for item in operator["communication"]
                ]
                self.assertIn("all_reduce", communication)
                self.assertIn("all_to_all", communication)

    def test_invalid_operator_returns_readable_error(self):
        config = json.loads((ROOT / "examples" / "single_chip_gqa.json").read_text(encoding="utf-8"))
        config["model"]["layer_pattern"][0]["attention"]["type"] = "typo_attention"
        request = Request(
            f"{self.base_url}/api/estimate",
            data=json.dumps(config).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as context:
            urlopen(request, timeout=3)
        self.assertEqual(context.exception.code, 400)
        error = json.loads(context.exception.read().decode("utf-8"))
        context.exception.close()
        self.assertIn("unknown operator type", error["error"])


if __name__ == "__main__":
    unittest.main()
