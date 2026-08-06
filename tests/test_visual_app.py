import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from transformer_modeling.visual_app.desktop import WINDOW_TITLE, build_parser as desktop_parser
from transformer_modeling.visual_app.server import FlowchartHandler, start_local_server


ROOT = Path(__file__).resolve().parents[1]


class VisualAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FlowchartHandler)
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

    def post_json(self, route, payload):
        request = Request(
            f"{self.base_url}{route}", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))

    def test_visual_entry_and_api_are_served(self):
        with urlopen(f"{self.base_url}/", timeout=3) as response:
            page = response.read().decode("utf-8")
        self.assertIn('id="sidebar-tabs"', page)
        self.assertIn('data-tab="model"', page)
        self.assertIn('js/app.js', page)
        schemas = self.get_json("/api/operator-schemas")
        self.assertEqual(schemas["schema_version"], 3)
        kv_dtype = schemas["operators"]["standard_attention"]["params"]["kv_cache_dtype"]
        self.assertEqual(kv_dtype["effect"], "performance")
        self.assertEqual(kv_dtype["inherit_from"], "model.dtype.kv_cache")
        self.assertTrue(kv_dtype["performance_impact"])
        self.assertEqual(schemas["operators"]["mhc"]["params"]["eps"]["effect"], "numerical")

    def test_preset_flowchart_round_trip_keeps_model_dimensions(self):
        resolved = self.post_json("/api/resolve-preset", {"preset_id": "glm-5.2"})
        config = resolved["config"]
        flowchart = self.post_json("/api/config-to-flowchart", {"config": config})
        self.assertEqual(flowchart["model_info"]["hidden_size"], 6144)
        self.assertTrue(all("residual_connections" not in node for node in flowchart["nodes"] if node["type"] == "layer_group"))
        flowchart["model_info"]["hidden_size"] = 5120
        flowchart["model_info"]["intermediate_size"] = 16384
        rebuilt = self.post_json("/api/flowchart-to-config", {
            "flowchart": flowchart, "base_config": config,
        })["config"]
        self.assertEqual(rebuilt["model"]["dimensions"]["hidden_size"], 5120)
        self.assertEqual(rebuilt["model"]["dimensions"]["intermediate_size"], 16384)

    def test_old_api_and_schema_are_not_available(self):
        for route in ("/api/local-devices", "/api/model-definitions/resolve"):
            with self.subTest(route=route), self.assertRaises(HTTPError) as caught:
                urlopen(f"{self.base_url}{route}", timeout=3)
            self.assertEqual(caught.exception.code, 404)
            caught.exception.close()
        expired = json.loads((ROOT / "examples" / "single_chip_gqa.json").read_text(encoding="utf-8"))
        expired["schema_version"] = 2
        with self.assertRaises(HTTPError) as caught:
            self.post_json("/api/estimate", expired)
        caught.exception.close()

    def test_desktop_shell_uses_an_ephemeral_local_server(self):
        self.assertEqual(desktop_parser().parse_args([]).debug, False)
        self.assertEqual(WINDOW_TITLE, "Transformer Modeling")
        server, url = start_local_server()
        try:
            self.assertTrue(url.startswith("http://127.0.0.1:"))
            with urlopen(url, timeout=3) as response:
                self.assertEqual(response.status, 200)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
