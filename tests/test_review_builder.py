import json
import re
import tempfile
import unittest
from pathlib import Path

try:
    import cv2
    import numpy as np
    import pandas as pd

    from octolyzer.review import __main__ as review_cli
    from octolyzer.review import builder, encode, flags, loader
except ImportError:
    np = None

MANIFEST = re.compile(r'<script type="application/json" id="octolyzer-manifest">(.*?)</script>', re.S)
PAYLOAD = re.compile(r'<script type="application/json" id="octolyzer-payload">(.*?)</script>', re.S)


def synthetic_data(n_scans=3, height=120, width=160):
    """A small, clean review data dictionary shaped like loader.load_review_data output."""
    rng = np.random.default_rng(0)
    columns = np.arange(width)
    base = 30 + 4 * np.sin(columns / 25.0)
    offsets = {"ILM": 0, "BM": 30, "CHORupper": 31, "CHORlower": 60}
    traces = {key: np.tile(base + offset, (n_scans, 1)) for key, offset in offsets.items()}
    vessels = np.zeros((n_scans, height, width), dtype=bool)
    vessels[:, 70:80, 40:90] = True
    slo_mask = np.zeros((96, 96), dtype=bool)
    slo_mask[40:44, 10:80] = True
    return {
        "filename": "Synthetic_1",
        "source": "Synthetic_1.vol",
        "scan_type": "Ppole",
        "location": "macular",
        "eye": "Right",
        "failed": False,
        "n_scans": n_scans,
        "bscans": rng.random((n_scans, height, width)),
        "bscan_shape": (height, width),
        "bscan_scale": (11.5, 3.9, 120.0),
        "quality": [30.0] * n_scans,
        "traces": traces,
        "layer_names": {},
        "vessels": vessels,
        "foveas": [None, [80.0, 30.0], None][:n_scans],
        "fovea_slice": 1,
        "roi_um": None,
        "acquisition": {"lines": [[10.0, 20.0 + 10 * i, 86.0, 20.0 + 10 * i] for i in range(n_scans)]},
        "slo": rng.random((96, 96)),
        "slo_scale": 11.5,
        "slo_masks": {"vessels": slo_mask, "artery": slo_mask},
        "slo_fovea": [48.0, 30.0],
        "optic_disc": None,
        "maps": [{"key": "retina", "unit": "um", "array": np.where(np.arange(96 * 96).reshape(96, 96) % 7 > 0, 250.0, -1.0)}],
        "grid": {"type": "etdrs", "centre": [48.0, 30.0], "angle": 0.0, "diameters_um": [1000, 3000, 6000]},
        "metadata": {"Filename": "Synthetic_1.vol", "eye": "Right", "bscan_type": "Ppole"},
        "measurements": {"etdrs_measurements": {"columns": ["map_name", "units", "central"], "rows": [["retina", "[um]", 250.0]]}},
        "log": ["Loaded a posterior pole scan."],
        "unavailable": [],
    }


@unittest.skipUnless(np is not None, "numpy, pandas and OpenCV are required for review tests")
class EncodingTests(unittest.TestCase):
    def test_traces_round_trip_with_missing_values(self):
        traces = np.array([[10.26, 11.0, np.nan, 0.0, 300.04], [1.0, 2.0, 3.0, 4.0, 5.0]])
        restored = encode.unpack_traces(encode.pack_traces(traces), traces.shape)
        self.assertTrue(np.isnan(restored[0, 2]) and np.isnan(restored[0, 3]))
        np.testing.assert_allclose(restored[0, [0, 1, 4]], [10.3, 11.0, 300.0], atol=1e-9)
        np.testing.assert_allclose(restored[1], traces[1])

    def test_uint16_packing_wraps_negative_steps(self):
        values = np.array([[5, 65535, 0, 12, 3]], dtype=np.uint16)
        np.testing.assert_array_equal(encode.unpack_uint16(encode.pack_uint16(values), values.shape), values)

    def test_map_keeps_values_and_marks_missing_pixels(self):
        values = np.full((64, 64), -1.0)
        values[16:48, 16:48] = 200.0
        packed = encode.pack_map(values, size=32)
        restored = encode.unpack_uint16(packed["data"], (32, 32)) / packed["scale"]
        self.assertEqual(restored[0, 0], 0)
        self.assertAlmostEqual(restored[16, 16], 200.0, delta=0.5)
        self.assertIsNone(encode.pack_map(np.full((8, 8), -1.0)))

    def test_images_and_masks_decode(self):
        image = np.linspace(0, 1, 64 * 64).reshape(64, 64)
        uri = encode.encode_image(image)
        self.assertTrue(uri.startswith(("data:image/webp;base64,", "data:image/png;base64,")))
        mask = np.zeros((20, 30), dtype=bool)
        mask[5:10, 5:10] = True
        import base64
        raw = np.frombuffer(base64.b64decode(encode.encode_mask(mask).split(",", 1)[1]), dtype=np.uint8)
        decoded = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(decoded[..., 3] > 0, mask)
        self.assertIsNone(encode.encode_mask(np.zeros((4, 4))))
        self.assertIsNone(encode.encode_mask(None))


@unittest.skipUnless(np is not None, "numpy, pandas and OpenCV are required for review tests")
class FlagTests(unittest.TestCase):
    def ids(self, data):
        return {flag["id"]: flag for flag in flags.compute_flags(data)}

    def test_clean_scan_has_no_flags(self):
        self.assertEqual(flags.compute_flags(synthetic_data()), [])

    def test_layer_crossing_reports_the_slice(self):
        data = synthetic_data()
        data["traces"]["BM"][2, 40:80] = data["traces"]["ILM"][2, 40:80] - 6
        found = self.ids(data)
        self.assertIn("layer_crossing", found)
        self.assertEqual(found["layer_crossing"]["slices"], [2])

    def test_gap_hole_and_jump(self):
        data = synthetic_data()
        data["traces"]["ILM"][0, :80] = np.nan
        data["traces"]["BM"][1, 50:60] = np.nan
        data["traces"]["CHORlower"][2, 100:] += 40
        found = self.ids(data)
        self.assertEqual(found["trace_gap"]["slices"], [0])
        self.assertEqual(found["trace_hole"]["slices"], [1])
        self.assertEqual(found["trace_jump"]["slices"], [2])

    def test_slice_outlier_in_volume(self):
        data = synthetic_data(n_scans=9)
        data["foveas"] = [None] * 9
        data["traces"]["CHORlower"][4] += 25
        self.assertEqual(self.ids(data)["slice_outlier"]["slices"], [4])

    def test_choroid_edge_quality_and_metadata(self):
        data = synthetic_data()
        data["traces"]["CHORlower"][1, 20:60] = data["bscan_shape"][0] - 1
        data["quality"] = [30.0, 9.0, 30.0]
        data["metadata"]["slo_missing_fovea"] = True
        data["log"].append("WARNING: AVR value exceeds 1, please check artery-vein segmentation.")
        found = self.ids(data)
        self.assertEqual(found["choroid_edge"]["slices"], [1])
        self.assertEqual(found["low_quality"]["slices"], [1])
        self.assertIn("slo_fovea_missing", found)
        self.assertIn("avr_high", found)
        self.assertNotIn("pipeline_warning", found)

    def test_missing_fovea_on_line_scan_and_failed_file(self):
        data = synthetic_data(n_scans=1)
        data.update(scan_type="H-line", foveas=[[0, 0]], fovea_slice=0)
        self.assertEqual(self.ids(data)["oct_fovea_missing"]["severity"], 3)
        failed = flags.compute_flags({"failed": True})
        self.assertEqual([flag["id"] for flag in failed], ["failed"])
        self.assertEqual(flags.severity_score(failed), 3)

    def test_unknown_layers_are_ordered_by_depth(self):
        traces = {"BM": np.full((1, 4), 50.0), "XYZ": np.full((1, 4), 20.0), "ILM": np.full((1, 4), 10.0)}
        self.assertEqual(flags.order_layers(traces), ["ILM", "XYZ", "BM"])


@unittest.skipUnless(np is not None, "numpy, pandas and OpenCV are required for review tests")
class BuilderTests(unittest.TestCase):
    def test_document_is_self_contained_with_manifest_at_the_top(self):
        html = builder.render_html(synthetic_data())
        head = html.encode("utf-8")[:builder.MANIFEST_WINDOW].decode("utf-8", errors="ignore")
        manifest = json.loads(MANIFEST.search(head).group(1))
        self.assertEqual(manifest["filename"], "Synthetic_1")
        self.assertEqual(manifest["source"], "Synthetic_1.vol")
        self.assertEqual(manifest["schema"], builder.SCHEMA_VERSION)
        self.assertEqual(manifest["n_scans"], 3)
        self.assertEqual(manifest["structures"], [
            "retinal_layers", "choroid_boundary", "choroid_vessels", "oct_fovea",
            "slo_vessels", "artery_vein", "slo_fovea", "image_quality"])
        for external in ("<link", "<script src", "@import", "url(http", 'src="http', "fetch("):
            self.assertNotIn(external, html)

    def test_payload_matches_the_data(self):
        data = synthetic_data()
        payload = json.loads(PAYLOAD.search(builder.render_html(data)).group(1))
        oct_payload = payload["oct"]
        self.assertEqual((oct_payload["width"], oct_payload["height"]), (160, 120))
        self.assertEqual(len(oct_payload["images"]), 3)
        self.assertEqual([layer["key"] for layer in oct_payload["layers"]], ["ILM", "BM", "CHORupper", "CHORlower"])
        restored = encode.unpack_traces(oct_payload["layers"][1]["data"], (3, 160), oct_payload["trace_scale"])
        np.testing.assert_allclose(restored, data["traces"]["BM"], atol=0.051)
        self.assertEqual(len(oct_payload["vessels"]), 3)
        self.assertAlmostEqual(oct_payload["slice_thickness"]["retina"][0], 30 * 3.9, delta=0.2)
        self.assertEqual(set(payload["slo"]["masks"]), {"vessels", "artery"})
        self.assertEqual(payload["maps"][0]["key"], "retina")
        self.assertEqual(payload["grid"]["diameters_um"], [1000, 3000, 6000])

    def test_script_breaking_text_is_escaped(self):
        data = synthetic_data()
        data["log"] = ["</script><script>alert(1)</script>", "<!-- comment"]
        html = builder.render_html(data)
        self.assertEqual(html.count("</script>"), 3)
        self.assertEqual(json.loads(PAYLOAD.search(html).group(1))["log"], data["log"])

    def test_optional_inputs_can_be_missing(self):
        data = synthetic_data()
        data.update(vessels=None, slo_masks={}, slo_fovea=None, maps=[], grid=None, traces={}, foveas=[None] * 3)
        html = builder.render_html(data)
        manifest = json.loads(MANIFEST.search(html).group(1))
        self.assertEqual(manifest["structures"], ["image_quality"])
        self.assertIn("no_layers", [flag["id"] for flag in manifest["flags"]])
        self.assertIsNone(json.loads(PAYLOAD.search(html).group(1))["oct"]["vessels"])

    def test_files_are_written_to_the_review_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            review_file = builder.write_review(synthetic_data(), output)
            self.assertEqual(review_file, output / "review" / "Synthetic_1.html")
            stub = builder.build_failed_stub(output / "Broken.vol", output, ["Failed to analyse Broken."])
            manifest = json.loads(MANIFEST.search(stub.read_text(encoding="utf-8")).group(1))
            self.assertTrue(manifest["failed"])
            self.assertEqual(manifest["source"], "Broken.vol")
            hub = builder.write_hub(output)
            hub_text = hub.read_text(encoding="utf-8")
            self.assertEqual(hub.name, "review.html")
            self.assertNotIn("__HUB_JS__", hub_text)
            self.assertNotIn("__THEME_CSS__", hub_text)
            # The hub must not mistake itself for a scan when it lists the folder.
            self.assertIsNone(MANIFEST.search(hub_text))
            self.assertEqual(list(output.glob("**/*.tmp")), [])


@unittest.skipUnless(np is not None, "numpy, pandas and OpenCV are required for review tests")
class LoaderTests(unittest.TestCase):
    def write_results(self, directory: Path, fname="Scan_1") -> Path:
        result_dir = directory / fname
        (result_dir / "thickness_maps").mkdir(parents=True)
        trace = pd.DataFrame([[0, 0.0, 12.0, 13.0], [1, 11.0, 12.0, np.nan]], columns=["scan_number", "0", "1", "2"])
        with pd.ExcelWriter(result_dir / f"{fname}_output.xlsx") as writer:
            pd.DataFrame([{"Filename": f"{fname}.vol", "eye": "Left", "bscan_type": "Ppole",
                           "slo_fovea_x": 5, "slo_fovea_y": 6, "avg_quality": np.float64(31.5)}]).to_excel(writer, sheet_name="metadata", index=False)
            pd.DataFrame([{"map_name": "retina", "units": "[um]", "central": 250}]).to_excel(writer, sheet_name="etdrs_measurements", index=False)
            (trace + [0, 20, 20, 20]).to_excel(writer, sheet_name="segmentations_BM", index=False)
            trace.to_excel(writer, sheet_name="segmentations_ILM", index=False)
            pd.DataFrame([{"key": "ILM", "layer": "Inner Limiting Membrane"}]).to_excel(writer, sheet_name="layer_keys", index=False)
        return result_dir

    def test_results_workbook_is_parsed(self):
        with tempfile.TemporaryDirectory() as directory:
            result_dir = self.write_results(Path(directory))
            results = loader.read_results(result_dir, "Scan_1")
        self.assertEqual(list(results["traces"]), ["ILM", "BM"])
        self.assertTrue(np.isnan(results["traces"]["ILM"][0, 0]) and np.isnan(results["traces"]["ILM"][1, 2]))
        self.assertEqual(results["traces"]["BM"][0, 1], 32.0)
        self.assertEqual(results["metadata"]["eye"], "Left")
        self.assertIsInstance(results["metadata"]["avg_quality"], float)
        self.assertEqual(results["measurements"]["etdrs_measurements"]["rows"], [["retina", "[um]", 250]])
        self.assertEqual(results["layer_names"], {"ILM": "Inner Limiting Membrane"})
        json.dumps(results["metadata"], allow_nan=False)

    def test_missing_results_raise(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                loader.read_results(Path(directory), "Missing")

    def test_saved_masks_and_maps_are_read(self):
        with tempfile.TemporaryDirectory() as directory:
            result_dir = self.write_results(Path(directory))
            avod = np.zeros((8, 8), dtype=np.uint8)
            avod[0, :] = 191
            avod[1, :] = 127
            avod[2, :] = 255
            cv2.imwrite(str(result_dir / "Scan_1_slo_avod_map.png"), avod)
            cv2.imwrite(str(result_dir / "Scan_1_slo_binary_map.png"), (avod > 0).astype(np.uint8) * 255)
            vessels = np.zeros((2, 6, 3))
            vessels[1, 2, 1] = 0.9
            vessels[1, 3, 1] = 0.2
            np.save(result_dir / "Scan_1_chorvessel_maps.npy", vessels)
            np.save(result_dir / "thickness_maps" / "Scan_1_retina_thickness_map.npy", np.ones((8, 8)))
            np.save(result_dir / "thickness_maps" / "Scan_1_choroid_CVI__map.npy", np.ones((8, 8)))

            masks = loader.read_slo_masks(result_dir, "Scan_1", (8, 8))
            self.assertEqual(set(masks), {"vessels", "artery", "vein", "disc"})
            self.assertTrue(masks["vein"][1].all() and not masks["vein"][0].any())
            self.assertEqual(loader.read_slo_masks(result_dir, "Scan_1", (9, 9)), {})
            read = loader.read_vessels(result_dir, "Scan_1", (2, 6, 3))
            self.assertEqual(int(read.sum()), 1)
            self.assertIsNone(loader.read_vessels(result_dir, "Scan_1", (1, 6, 3)))
            measurements = loader.read_results(result_dir, "Scan_1")["measurements"]
            maps = loader.read_maps(result_dir, "Scan_1", measurements)
            self.assertEqual([(item["key"], item["unit"]) for item in maps], [("retina", "um"), ("choroid_CVI", "")])

    def test_fovea_slice_is_the_closest_acquisition_line(self):
        lines = np.array([[[0, 10], [100, 10]], [[0, 20], [100, 20]], [[0, 30], [100, 30]]], dtype=float)
        self.assertEqual(loader._fovea_slice(lines, [50, 22]), 1)
        self.assertEqual(loader._fovea_slice(lines, None), 1)

    def test_run_configuration_is_read_for_retroactive_builds(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "configuration_used.txt").write_text(
                "# comment: ignored\nanalysis_directory: C:\\scans\\study\npreprocess_bscans: 0\n", encoding="utf-8")
            config = review_cli.read_run_configuration(output)
            self.assertEqual(config, {"analysis_directory": "C:\\scans\\study", "preprocess_bscans": "0"})
            self.assertEqual(review_cli.read_run_configuration(output / "missing"), {})
            with self.assertRaises(FileNotFoundError):
                review_cli.build_all(output, analysis_directory=output / "missing", verbose=False)


if __name__ == "__main__":
    unittest.main()
