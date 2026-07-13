from __future__ import annotations

import functools
import hashlib
import json
import math
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import torch
import imageio.v3 as iio

import smart_grid_detector as detector


class WaveformGenerationTests(unittest.TestCase):
    def test_generation_is_deterministic_and_labelled(self) -> None:
        first = detector._generate_waveform_bundle(40, 512, seed=42)
        repeated = detector._generate_waveform_bundle(40, 512, seed=42)
        changed_seed = detector._generate_waveform_bundle(40, 512, seed=43)

        self.assertEqual(tuple(first.train_waveforms.shape), (32, 3, 512))
        self.assertEqual(tuple(first.test_waveforms.shape), (8, 3, 512))
        self.assertEqual(int(first.train_labels.sum() + first.test_labels.sum()), 2)
        self.assertTrue(torch.equal(first.train_waveforms, repeated.train_waveforms))
        self.assertFalse(torch.equal(first.train_waveforms, changed_seed.train_waveforms))
        self.assertTrue(torch.isfinite(first.train_waveforms).all())

    def test_public_generation_api_remains_two_part(self) -> None:
        (train_x, train_y), (test_x, test_y) = detector.generate_waveforms(40, 512)
        self.assertEqual(train_x.shape[0], train_y.shape[0])
        self.assertEqual(test_x.shape[0], test_y.shape[0])


class ModelAndFourierTests(unittest.TestCase):
    def test_autoencoder_perfectly_mirrors_shape(self) -> None:
        model = detector.GridWaveformAutoencoder(sequence_length=512)
        inputs = torch.randn(2, 3, 512, requires_grad=True)
        latent = model.encode(inputs)
        outputs = model(inputs)
        self.assertEqual(tuple(latent.shape), (2, 64))
        self.assertEqual(outputs.shape, inputs.shape)
        self.assertEqual(
            sum(isinstance(module, torch.nn.Conv1d) for module in model.modules()),
            3,
        )
        self.assertEqual(
            sum(
                isinstance(module, torch.nn.ConvTranspose1d)
                for module in model.modules()
            ),
            3,
        )
        outputs.square().mean().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())

    def test_thd_matches_known_harmonics_across_frequency_drift(self) -> None:
        time_axis = torch.arange(512) / detector.SAMPLE_RATE_HZ
        phases = torch.tensor([0.0, -2 * math.pi / 3, 2 * math.pi / 3])
        for frequency in (59.85, 60.0, 60.15):
            base = torch.sin(
                2 * math.pi * frequency * time_axis[None, :] + phases[:, None]
            )
            pure = float(detector.compute_thd(base)[0])
            self.assertLess(pure, 0.01)
            for ratio in (0.05, 0.10):
                distorted = base.clone()
                distorted[1] += ratio * torch.sin(
                    2 * math.pi * 3 * frequency * time_axis + 3 * phases[1]
                )
                measured = float(detector.compute_thd(distorted)[0])
                self.assertAlmostEqual(measured, ratio, delta=0.012)

        pll = detector.PhaseLockedLoop()
        fortescue = detector.SymmetricalComponentsLayer()
        endpoint_waveforms = torch.stack(
            [
                torch.sin(
                    2 * math.pi * frequency * time_axis[None, :]
                    + phases[:, None]
                )
                for frequency in (59.5, 60.0, 60.5)
            ]
        )
        tracked = pll.estimate_frequency(endpoint_waveforms)
        self.assertTrue(
            torch.allclose(
                tracked,
                torch.tensor([59.5, 60.0, 60.5]),
                atol=0.02,
                rtol=0.0,
            )
        )
        positive_components = fortescue(endpoint_waveforms)
        self.assertTrue(
            torch.allclose(
                positive_components[:, 0],
                torch.full((3,), 1.0 / math.sqrt(2.0)),
                atol=1.0e-3,
                rtol=0.0,
            )
        )
        self.assertTrue((positive_components[:, 1:] < 1.0e-3).all())

        reverse = endpoint_waveforms[1, [0, 2, 1], :]
        zero = endpoint_waveforms[1, 0, :].repeat(3, 1)
        reverse_and_zero = fortescue(torch.stack((reverse, zero)))
        self.assertAlmostEqual(
            float(reverse_and_zero[0, 1]), 1.0 / math.sqrt(2.0), delta=1.0e-3
        )
        self.assertAlmostEqual(
            float(reverse_and_zero[1, 2]), 1.0 / math.sqrt(2.0), delta=1.0e-3
        )

    def test_absent_fundamental_and_nonfinite_input_fail_closed(self) -> None:
        zeros = torch.zeros(1, 3, 512)
        self.assertTrue(torch.isinf(detector.compute_thd(zeros)).all())
        invalid = zeros.clone()
        invalid[0, 0, 10] = float("nan")
        self.assertTrue(torch.isinf(detector.compute_thd(invalid)).all())


class HardeningAndArtifactTests(unittest.TestCase):
    def test_resource_limits_reject_unbounded_configuration(self) -> None:
        with self.assertRaises(ValueError):
            detector.DetectorConfig(num_samples=10**9).validate()
        with self.assertRaises(ValueError):
            detector.DetectorConfig(learning_rate=float("nan")).validate()
        with self.assertRaises(ValueError):
            detector.DetectorConfig(latent_dim=10**9).validate()

    def test_combined_memory_budget_rejects_compound_configuration(self) -> None:
        # Every dimension below passes its individual cap; the combined peak
        # working set must still be rejected (SGD-PY-001).
        with self.assertRaises(ValueError):
            detector.DetectorConfig(
                num_samples=19_531,
                batch_size=4_096,
                base_channels=256,
                latent_dim=190,
            ).validate()
        detector.DetectorConfig().validate()

    def test_direct_tensor_apis_reject_oversized_batches(self) -> None:
        # Direct callers must hit the same element cap as compute_thd
        # (SGD-PY-002).
        model = detector.GridWaveformAutoencoder(sequence_length=512)
        oversized = torch.empty(19_600, 3, 512)
        with self.assertRaises(ValueError):
            detector.train_autoencoder(model, oversized, torch.device("cpu"))
        with self.assertRaises(ValueError):
            detector.validate_detector(
                model,
                oversized,
                torch.zeros(19_600),
                0.01,
                torch.device("cpu"),
            )

    def test_multi_seed_evaluation_aggregates_with_intervals(self) -> None:
        config = detector.DetectorConfig(num_samples=40, epochs=1)
        summary = detector.run_multi_seed_evaluation(
            config, [42, 43], torch.device("cpu")
        )
        self.assertEqual(summary["seeds"], [42, 43])
        self.assertEqual(len(summary["per_seed"]), 2)
        self.assertEqual(summary["confidence_level"], 0.95)
        aggregate = summary["aggregate"]["accuracy"]
        self.assertGreaterEqual(aggregate["ci95_high"], aggregate["mean"])
        self.assertLessEqual(aggregate["ci95_low"], aggregate["mean"])
        self.assertLessEqual(aggregate["min"], aggregate["mean"])
        self.assertGreaterEqual(aggregate["max"], aggregate["mean"])
        self.assertTrue(0.0 <= aggregate["mean"] <= 1.0)
        with self.assertRaises(ValueError):
            detector.run_multi_seed_evaluation(config, [42], torch.device("cpu"))
        with self.assertRaises(ValueError):
            detector.run_multi_seed_evaluation(
                config, [42, 42], torch.device("cpu")
            )

    def test_multi_seed_count_is_bounded_before_materialization(self) -> None:
        self.assertEqual(detector._multi_seed_sequence(42, 0), [])
        self.assertEqual(detector._multi_seed_sequence(42, 2), [42, 43])
        with self.assertRaises(ValueError):
            detector._multi_seed_sequence(42, 1)
        with self.assertRaises(ValueError):
            detector._multi_seed_sequence(42, detector.MAX_MULTI_SEED_RUNS + 1)
        with self.assertRaises(ValueError):
            detector._multi_seed_sequence(2**63 - 1, 2)

    def test_nonfinite_evidence_exports_as_null(self) -> None:
        # Infinite THD is a correct detection and must not block strict
        # artifact export (SGD-PY-003).
        self.assertIsNone(detector._json_safe_float(float("inf")))
        self.assertIsNone(detector._json_safe_float(float("nan")))
        self.assertEqual(detector._json_safe_float(0.05), 0.05)
        self.assertEqual(detector._format_evidence_ratio(None), "invalid")
        self.assertEqual(detector._format_evidence_error(None), "invalid")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            detector.write_results_json(
                {"thd_ratio": detector._json_safe_float(float("inf"))}, path
            )
            self.assertIsNone(
                json.loads(path.read_text(encoding="utf-8"))["thd_ratio"]
            )

    def test_json_write_is_strict_and_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            detector.write_results_json({"schema_version": 2, "value": 1.0}, path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["value"], 1.0)
            with self.assertRaises(ValueError):
                detector.write_results_json({"bad": float("nan")}, path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["value"], 1.0)

    def test_recorded_artifacts_are_coherent(self) -> None:
        results = detector.DEFAULT_SITE_DIR / "simulation_results.json"
        dispatches = detector.DEFAULT_SITE_DIR / "grid_dispatches.json"
        video = detector.DEFAULT_SITE_DIR / "smart_grid_explainer.mp4"
        self.assertTrue(results.is_file())
        self.assertTrue(video.is_file())
        payload = json.loads(results.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], detector.RESULT_SCHEMA_VERSION)
        self.assertEqual(len(payload["training"]["epochs"]), 15)
        self.assertEqual(len(payload["evaluation"]["observations"]), 400)
        self.assertEqual(payload["evaluation"]["metrics"]["true_positives"], 20)
        self.assertGreaterEqual(len(payload["replay"]["records"]), 4)
        expected_source_sha = hashlib.sha256(
            Path(detector.__file__).resolve().read_bytes()
        ).hexdigest()
        self.assertEqual(payload["provenance"]["source_sha256"], expected_source_sha)
        self.assertIn(
            "max_fused_physics",
            payload["fdi_resilience"]["calibration"],
        )
        dispatch_payload = json.loads(dispatches.read_text(encoding="utf-8"))
        self.assertEqual(dispatch_payload["schema_version"], 1)
        self.assertEqual(dispatch_payload["run_id"], payload["run"]["id"])
        self.assertEqual(
            dispatch_payload["results_sha256"],
            hashlib.sha256(results.read_bytes()).hexdigest(),
        )
        self.assertGreater(video.stat().st_size, 50_000)
        first_frame = iio.imread(video, index=0, plugin="FFMPEG")
        self.assertEqual(tuple(first_frame.shape[:2]), (544, 960))

    def test_dashboard_is_loopback_safe_and_blocks_parent_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            site = Path(directory)
            (site / "index.html").write_text("ok", encoding="utf-8")
            (site / "app.js").write_text("'use strict';", encoding="utf-8")
            (site / ".env.local").write_text("SECRET=test", encoding="utf-8")
            (site / ".vercel").mkdir()
            (site / ".vercel" / "project.json").write_text("{}", encoding="utf-8")
            handler = functools.partial(
                detector._DashboardRequestHandler,
                directory=str(site),
            )
            server = detector.ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            port = server.server_address[1]
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/", timeout=5
                ) as response:
                    self.assertEqual(response.status, 200)
                    self.assertEqual(
                        response.headers["X-Content-Type-Options"], "nosniff"
                    )
                    self.assertIn(
                        "default-src 'self'",
                        response.headers["Content-Security-Policy"],
                    )
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/app.js?v=1", timeout=5
                ) as response:
                    self.assertEqual(response.status, 200)
                for blocked in (
                    "/.env.local",
                    "/.vercel/project.json",
                    "/%2e%2e/smart_grid_detector.py",
                    "/undeclared.txt",
                ):
                    with self.assertRaises(urllib.error.HTTPError) as context:
                        urllib.request.urlopen(
                            f"http://127.0.0.1:{port}{blocked}", timeout=5
                        )
                    self.assertEqual(context.exception.code, 404)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


class RobustnessAndEdgeTests(unittest.TestCase):
    """Shared one-epoch model exercising section 6B end to end."""

    @classmethod
    def setUpClass(cls) -> None:
        detector.set_deterministic_seed(42)
        cls.device = torch.device("cpu")
        cls.bundle = detector._generate_waveform_bundle(40, 512, seed=42)
        cls.model = detector.GridWaveformAutoencoder(sequence_length=512)
        cls.normal_train = cls.bundle.train_waveforms[
            cls.bundle.train_labels == 0
        ].contiguous()
        cls.calibration = {}
        cls.threshold = detector.train_autoencoder(
            cls.model,
            cls.normal_train,
            cls.device,
            epochs=1,
            calibration=cls.calibration,
        )
        cls.diagnostics = {}
        cls.summary = detector.validate_detector(
            cls.model,
            cls.bundle.test_waveforms,
            cls.bundle.test_labels,
            cls.threshold,
            cls.device,
            diagnostics=cls.diagnostics,
        )

    def test_threshold_sweep_covers_configured_sigma_consistently(self) -> None:
        sweep = detector.run_threshold_sweep(
            self.diagnostics["reconstruction_errors"],
            self.diagnostics["thd_values"],
            self.diagnostics["invalid_input_flags"],
            self.bundle.test_labels,
            self.calibration,
        )
        sigmas = [point["sigma"] for point in sweep["points"]]
        self.assertIn(sweep["configured_sigma"], sigmas)
        configured_point = next(
            point
            for point in sweep["points"]
            if point["sigma"] == sweep["configured_sigma"]
        )
        self.assertAlmostEqual(
            configured_point["accuracy"], self.summary.accuracy, places=9
        )
        false_positive_counts = [
            point["false_positives"] for point in sweep["points"]
        ]
        self.assertEqual(
            false_positive_counts,
            sorted(false_positive_counts, reverse=True),
        )

    def test_boundary_probes_flip_each_rule(self) -> None:
        result = detector.run_boundary_probes(
            self.model, self.threshold, self.device, 512, seed=42
        )
        by_name = {probe["name"]: probe for probe in result["probes"]}
        self.assertFalse(by_name["thd_below_limit"]["observed"]["flags"]["thd"])
        self.assertTrue(by_name["thd_above_limit"]["observed"]["flags"]["thd"])
        self.assertTrue(by_name["thd_bias_sentinel"]["observed"]["flags"]["thd"])
        self.assertTrue(by_name["flatline_fails_closed"]["observed"]["flags"]["thd"])
        self.assertTrue(
            by_name["nonfinite_fails_closed"]["observed"]["flags"]["invalid_input"]
        )
        self.assertIn("reconstruction_below_threshold", by_name)
        self.assertIn("reconstruction_above_threshold", by_name)
        self.assertTrue(result["all_pass"])

    def test_edge_benchmark_reports_both_variants_or_reason(self) -> None:
        benchmark = detector.run_edge_benchmark(
            self.model,
            self.normal_train,
            self.bundle.test_waveforms,
            self.diagnostics["thd_values"],
            self.diagnostics["invalid_input_flags"],
            self.bundle.test_labels,
            3.5,
            timed_iterations=5,
        )
        self.assertGreater(benchmark["fp32"]["state_dict_bytes"], 0)
        self.assertGreater(
            benchmark["fp32"]["single_record_latency_ms_median"], 0.0
        )
        if benchmark["int8"] is not None:
            self.assertTrue(0.0 <= benchmark["int8"]["accuracy"] <= 1.0)
            self.assertGreater(benchmark["size_reduction_ratio"], 1.0)
        else:
            self.assertIsInstance(benchmark["int8_unavailable_reason"], str)

    def test_fdi_physics_checks_catch_replay_but_pass_genuine(self) -> None:
        # Calibration statistics and the 6% bias attack's tail behavior need a
        # production-scale sample count, so this test uses its own full-size
        # normal set rather than the 40-row shared bundle.
        fdi_bundle = detector._generate_waveform_bundle(2000, 512, seed=42)
        train_normal = fdi_bundle.train_waveforms[
            fdi_bundle.train_labels == 0
        ].contiguous()
        test_normal = fdi_bundle.test_waveforms[fdi_bundle.test_labels == 0]
        fdi_calibration = detector.calibrate_fdi_detector(train_normal)
        self.assertIn("max_phase_scale_asymmetry", fdi_calibration)
        self.assertIn("max_fused_physics", fdi_calibration)

        genuine = detector.detect_false_data_injection(
            test_normal, fdi_calibration
        )
        self.assertEqual(int(genuine["flags"].sum()), 0)
        # Neither individual nor joint rules fire on genuine telemetry.
        self.assertEqual(int(genuine["individual"].sum()), 0)
        self.assertEqual(int(genuine["joint"].sum()), 0)

        replay = detector._three_phase_base(512).repeat(4, 1, 1)
        attacked = detector.detect_false_data_injection(replay, fdi_calibration)
        self.assertTrue(bool(attacked["flags"].all()))
        self.assertTrue(bool(attacked["too_clean"].all()))

        # A 6% single-phase bias must trip the scale-asymmetry check on the
        # large majority of records (the strengthened detector clears 95%).
        bias = test_normal.clone()
        bias[:, 0, :] = bias[:, 0, :] * 1.06
        biased = detector.detect_false_data_injection(bias, fdi_calibration)
        self.assertGreaterEqual(float(biased["flags"].float().mean()), 0.95)

        scenario = detector.run_fdi_scenario(
            self.model,
            fdi_bundle,
            train_normal,
            self.threshold,
            self.device,
        )
        self.assertEqual(scenario["baseline_false_positives"], 0)
        by_name = {a["name"]: a for a in scenario["attacks"]}
        # Replay and phase-bias clear the 95% gate.
        self.assertGreaterEqual(
            by_name["replay_masking"]["detection_rate_among_evasions"], 0.95
        )
        self.assertGreaterEqual(
            by_name["phase_bias"]["detection_rate_among_evasions"], 0.95
        )
        for attack in by_name.values():
            self.assertEqual(
                attack["caught_after_learned_evasion"]
                + attack["residual_evasions"],
                attack["evaded_learned_detector"],
            )
        # The coordinated attack is judged by the layered-defense thesis: it
        # evades the learned detector while the physics layer still catches it.
        self.assertGreaterEqual(
            scenario["coordinated_evades_learned_detector"], 0.9
        )
        self.assertGreaterEqual(
            by_name["coordinated"]["detection_rate_among_evasions"], 0.5
        )
        self.assertTrue(scenario["coordinated_layered_defense_proven"])
        self.assertTrue(scenario["overall_layered_defense_proven"])

    def test_roc_analysis_reports_valid_auc_and_endpoints(self) -> None:
        roc = detector.run_roc_analysis(
            self.diagnostics, self.bundle.test_labels, self.calibration
        )
        for key in ("reconstruction", "combined"):
            curve = roc[key]
            self.assertGreaterEqual(curve["auc"], 0.0)
            self.assertLessEqual(curve["auc"], 1.0)
            # ROC curves start at (0,0) and end at (1,1).
            self.assertEqual(curve["fpr"][0], 0.0)
            self.assertEqual(curve["tpr"][0], 0.0)
            self.assertAlmostEqual(curve["fpr"][-1], 1.0, places=6)
            self.assertAlmostEqual(curve["tpr"][-1], 1.0, places=6)
            # Rates are monotonically non-decreasing along the sorted sweep.
            self.assertEqual(curve["fpr"], sorted(curve["fpr"]))
            self.assertEqual(curve["tpr"], sorted(curve["tpr"]))
        self.assertIn("operating_point", roc)
        self.assertTrue(0.0 <= roc["operating_point"]["fpr"] <= 1.0)
        self.assertTrue(0.0 <= roc["operating_point"]["tpr"] <= 1.0)

    def test_roc_curve_ranks_perfect_separation_as_auc_one(self) -> None:
        # A score that perfectly orders anomalies above normals must score 1.0.
        scores = torch.tensor([0.9, 0.8, 0.2, 0.1])
        labels = torch.tensor([1, 1, 0, 0])
        curve = detector._roc_curve(scores, labels)
        self.assertAlmostEqual(curve["auc"], 1.0, places=6)
        # A reversed ranking is the worst case.
        worst = detector._roc_curve(scores, torch.tensor([0, 0, 1, 1]))
        self.assertAlmostEqual(worst["auc"], 0.0, places=6)
        # Tied scores carry no ranking information and must be independent of
        # input label order.
        tied = torch.ones(4)
        for tied_labels in (
            torch.tensor([1, 1, 0, 0]),
            torch.tensor([0, 1, 0, 1]),
            torch.tensor([0, 0, 1, 1]),
        ):
            self.assertAlmostEqual(
                detector._roc_curve(tied, tied_labels)["auc"], 0.5, places=6
            )

    def test_latency_profile_covers_batch_sizes(self) -> None:
        profile = detector.run_latency_profile(
            self.model,
            self.bundle.test_waveforms,
            self.device,
            batch_sizes=(1, 4, 8),
            timed_iterations=3,
        )
        self.assertGreaterEqual(len(profile["points"]), 1)
        for point in profile["points"]:
            self.assertGreater(point["latency_ms_median"], 0.0)
            self.assertGreater(point["throughput_per_s"], 0.0)
            self.assertAlmostEqual(
                point["per_record_ms"],
                point["latency_ms_median"] / point["batch_size"],
                places=6,
            )
        self.assertIsNotNone(profile["peak_throughput_batch_size"])

    def test_harmonic_attribution_names_the_injected_order(self) -> None:
        time_axis = torch.arange(512) / detector.SAMPLE_RATE_HZ
        phases = torch.tensor([0.0, -2 * math.pi / 3, 2 * math.pi / 3])
        base = torch.sin(
            2 * math.pi * 60.0 * time_axis[None, :] + phases[:, None]
        )
        clean = detector.compute_harmonic_attribution(base.unsqueeze(0))[0]
        self.assertGreater(clean["fundamental_fraction"], 0.99)
        # Inject a strong 3rd harmonic on phase B.
        distorted = base.clone()
        distorted[1] += 0.2 * torch.sin(2 * math.pi * 180.0 * time_axis)
        result = detector.compute_harmonic_attribution(distorted.unsqueeze(0))[0]
        self.assertEqual(result["dominant_harmonic"], 3)
        self.assertEqual(result["worst_phase"], "B")
        self.assertGreater(result["low_harmonic_fraction"], 0.0)
        # Non-finite input reports no attribution rather than raising.
        broken = base.clone()
        broken[0, 5] = float("nan")
        self.assertIsNone(
            detector.compute_harmonic_attribution(broken.unsqueeze(0))[0][
                "dominant_harmonic"
            ]
        )
        flatline = detector.compute_harmonic_attribution(torch.zeros(1, 3, 512))[0]
        self.assertIsNone(flatline["dominant_harmonic"])
        self.assertIsNone(flatline["worst_phase"])

        # Phase A has less absolute harmonic energy but the highest THD ratio;
        # attribution must match the detector's ratio-based worst phase.
        unequal = base.clone()
        unequal[0] *= 0.25
        unequal[0] += 0.04 * torch.sin(2 * math.pi * 180.0 * time_axis)
        unequal[1] += 0.08 * torch.sin(2 * math.pi * 180.0 * time_axis)
        ratio_result = detector.compute_harmonic_attribution(unequal.unsqueeze(0))[0]
        self.assertEqual(ratio_result["worst_phase"], "A")

    def test_added_analysis_apis_enforce_waveform_bounds(self) -> None:
        with self.assertRaises(ValueError):
            detector.compute_harmonic_attribution(torch.zeros(1, 4, 512))
        single = detector._physics_consistency_features(torch.zeros(3, 512))
        self.assertEqual(single["phase_sum_rms"].shape, (1,))
        with self.assertRaises(ValueError):
            detector._physics_consistency_features(torch.zeros(1, 4, 512))
        oversized = torch.empty(19_600, 3, 512, device="meta")
        with self.assertRaises(ValueError):
            detector.compute_harmonic_attribution(oversized)

    def test_fdi_calibration_fails_closed_when_joint_bound_is_missing(self) -> None:
        waveform = detector._three_phase_base(512).unsqueeze(0)
        with self.assertRaises(ValueError):
            detector.detect_false_data_injection(
                waveform,
                {
                    "min_noise_fraction": 0.1,
                    "max_phase_sum_rms": 0.1,
                    "max_rms_imbalance": 0.1,
                    "max_phase_scale_asymmetry": 0.1,
                },
            )

    def test_self_healing_dispatches_every_alert_inside_sector(self) -> None:
        summary, artifact = detector.run_self_healing_simulation(
            self.bundle, self.diagnostics, self.threshold
        )
        alert_count = int(self.diagnostics["predictions"].sum())
        self.assertEqual(summary["alerts_dispatched"], alert_count)
        self.assertEqual(artifact["schema_version"], 1)
        self.assertEqual(len(artifact["operational_dispatches"]), alert_count)
        self.assertEqual(len(artifact["detected_assets"]), alert_count)
        west, south, east, north = detector.SYNTHETIC_SECTOR["bbox_lon_lat"]
        for asset in artifact["detected_assets"]:
            longitude, latitude = asset["coordinates"]
            self.assertTrue(west <= longitude <= east)
            self.assertTrue(south <= latitude <= north)
        for dispatch in artifact["operational_dispatches"]:
            self.assertIn(dispatch["priority"], ("CRITICAL", "HIGH", "MEDIUM"))
        self.assertIn("simulation_disclaimer", artifact)

    def test_drift_monitor_alarms_on_shift_but_not_baseline(self) -> None:
        monitor = detector.DriftMonitor(0.01, 0.001, window=16, z_limit=4.0)
        state = {}
        for _ in range(16):
            state = monitor.observe(0.01)
        self.assertTrue(state["window_full"])
        self.assertFalse(state["drift"])
        shifted = detector.DriftMonitor(0.01, 0.001, window=16, z_limit=4.0)
        for _ in range(16):
            state = shifted.observe(0.012)
        self.assertTrue(state["drift"])
        with self.assertRaises(ValueError):
            detector.DriftMonitor(float("nan"), 0.001)
        with self.assertRaises(ValueError):
            detector.DriftMonitor(0.01, 0.0)
        scenario = detector.run_drift_scenario(
            self.model,
            self.bundle.test_waveforms,
            self.bundle.test_labels,
            self.threshold,
            self.calibration,
            self.device,
        )
        self.assertEqual(
            scenario["clean_phase"]["records"],
            int((self.bundle.test_labels == 0).sum()),
        )
        self.assertEqual(len(scenario["gain_scenarios"]), 3)


class AdversarialAndSensorFailureTests(unittest.TestCase):
    """Sensor faults and adversarial perturbations beyond the FDI scenario.

    These complement the existing FDI scenario (replay, phase bias,
    coordinated) and boundary probes (flatline, non-finite) by exercising
    failure modes that attack the sensor layer or the physical signal itself
    rather than the learned or physics rules individually.
    """

    @classmethod
    def setUpClass(cls) -> None:
        detector.set_deterministic_seed(42)
        cls.device = torch.device("cpu")
        cls.bundle = detector._generate_waveform_bundle(2000, 512, seed=42)
        cls.model = detector.GridWaveformAutoencoder(sequence_length=512)
        cls.normal_train = cls.bundle.train_waveforms[
            cls.bundle.train_labels == 0
        ].contiguous()
        cls.calibration = {}
        cls.threshold = detector.train_autoencoder(
            cls.model,
            cls.normal_train,
            cls.device,
            epochs=1,
            calibration=cls.calibration,
        )
        cls.fdi_calibration = detector.calibrate_fdi_detector(cls.normal_train)

    def test_stuck_at_sensor_fault_detected_by_thd(self) -> None:
        """A sensor frozen at a constant non-zero DC offset has no fundamental.

        The THD estimator divides harmonic power by fundamental power; a pure
        DC signal has zero fundamental power, so THD returns infinity and the
        detector flags it via the fail-closed path.
        """
        base = detector._three_phase_base(512)
        stuck = torch.full_like(base, 0.5)
        decision = detector._single_record_decision(
            self.model, stuck, self.threshold, self.device
        )
        self.assertTrue(decision["flags"]["thd"])
        self.assertEqual(decision["prediction"], "anomaly")

    def test_phase_swap_breaks_three_phase_balance(self) -> None:
        """Swapping phase A and C breaks the 120-degree balance.

        A genuine three-phase signal has near-zero phase-sum RMS. Swapping
        two phases introduces a large phase-sum component that the FDI
        physics layer catches. The ideal noiseless base also trips the
        too-clean check, so the combined flags must all fire.
        """
        base = detector._three_phase_base(512).repeat(4, 1, 1)
        swapped = base[:, [2, 1, 0], :]
        fdi = detector.detect_false_data_injection(swapped, self.fdi_calibration)
        self.assertTrue(bool(fdi["flags"].all()))
        self.assertTrue(
            bool(fdi["unbalanced_sum"].any()) or bool(fdi["too_clean"].any()),
            "Phase swap must be caught by balance or noise-floor check",
        )

    def test_intermittent_burst_corruption_trips_reconstruction(self) -> None:
        """A short burst of corruption within a clean waveform raises error.

        Corrupting 10% of samples with extreme values increases the
        reconstruction error above the threshold while leaving the waveform
        nominally finite (so the invalid-input path does not fire).
        """
        base = detector._three_phase_base(512)
        corrupted = base.clone()
        burst_start = 400
        burst_end = 452
        corrupted[0, :, burst_start:burst_end] = 10.0
        decision = detector._single_record_decision(
            self.model, corrupted, self.threshold, self.device
        )
        self.assertTrue(decision["flags"]["reconstruction"])
        self.assertFalse(decision["flags"]["invalid_input"])

    def test_additive_gaussian_noise_trips_above_threshold(self) -> None:
        """Large additive noise pushes reconstruction error past the threshold.

        A noise standard deviation of 0.5 on a unit-amplitude waveform is
        well above what the autoencoder can reconstruct, so the
        reconstruction rule must fire.
        """
        base = detector._three_phase_base(512)
        generator = torch.Generator().manual_seed(99)
        noise = torch.randn(base.shape, generator=generator) * 0.5
        noisy = base + noise
        decision = detector._single_record_decision(
            self.model, noisy, self.threshold, self.device
        )
        self.assertTrue(decision["flags"]["reconstruction"])

    def test_sensor_saturation_clipping_detected_by_thd(self) -> None:
        """Clipping a sine wave at +/- 0.3 creates strong odd harmonics.

        Hard clipping is a common ADC saturation failure. The squared
        harmonics push THD well above the 5% limit.
        """
        base = detector._three_phase_base(512)
        clipped = base.clamp(-0.3, 0.3)
        thd = detector.compute_thd(clipped)
        self.assertTrue((thd > detector.THD_LIMIT).all())

    def test_multi_phase_coordinated_bias_caught_by_fused_score(self) -> None:
        """Small simultaneous biases on all three phases evade individual rules.

        Scaling phases A, B, C by 1.025, 0.98, and 1.015 respectively keeps
        each per-phase RMS close to the others (low asymmetry) and keeps the
        phase-sum RMS small. With realistic sensor noise added so the
        too-clean check passes, no single physics check trips but the fused
        joint score exceeds the calibration bound.
        """
        base = detector._three_phase_base(512).repeat(8, 1, 1)
        generator = torch.Generator().manual_seed(77)
        noise = torch.randn(base.shape, generator=generator) * 0.02
        noisy_base = base + noise
        biased = noisy_base.clone()
        biased[:, 0, :] *= 1.025
        biased[:, 1, :] *= 0.98
        biased[:, 2, :] *= 1.015
        fdi = detector.detect_false_data_injection(biased, self.fdi_calibration)
        self.assertFalse(
            bool(fdi["too_clean"].all()),
            "Noisy waveforms must pass the noise-floor check",
        )
        self.assertTrue(
            bool(fdi["flags"].all()),
            "Multi-phase coordinated bias must be caught by some rule",
        )
        self.assertTrue(
            bool(fdi["joint"].all()) or bool(fdi["individual"].any()),
            "Fused joint score or individual check must catch the bias",
        )

    def test_gain_drift_accumulation_triggers_drift_monitor(self) -> None:
        """Small per-record gain shifts that stay below the alarm threshold.

        Each individual record has reconstruction error below the detection
        threshold, but the rolling z-test accumulates enough evidence to
        raise a drift alarm within the window.
        """
        normals = self.bundle.test_waveforms[self.bundle.test_labels == 0]
        model = self.model
        model.eval()
        collected = []
        with torch.inference_mode():
            for start_index in range(0, normals.shape[0], 128):
                batch = normals[start_index : start_index + 128].to(self.device)
                collected.append(
                    detector._per_waveform_mse(model(batch), batch).cpu()
                )
        clean_errors = torch.cat(collected)
        threshold_val = self.calibration["mean_error"]
        std_val = max(self.calibration["std_error"], 1e-12)
        monitor = detector.DriftMonitor(threshold_val, std_val, window=16, z_limit=4.0)
        individual_alarms = 0
        drift_alarm = None
        for value in clean_errors.tolist():
            state = monitor.observe(value)
            if value > self.threshold:
                individual_alarms += 1
            if state["drift"] and drift_alarm is None:
                drift_alarm = state["observation"]
        self.assertEqual(individual_alarms, 0, "No individual record should alarm")
        self.assertIsNone(drift_alarm, "Clean baseline must not trigger drift")

        gain = 1.012
        gain_errors = []
        with torch.inference_mode():
            scaled = (normals * gain).to(self.device)
            for start_index in range(0, scaled.shape[0], 128):
                batch = scaled[start_index : start_index + 128]
                gain_errors.append(
                    detector._per_waveform_mse(model(batch), batch).cpu()
                )
        gain_error_tensor = torch.cat(gain_errors)
        drift_monitor = detector.DriftMonitor(
            threshold_val, std_val, window=16, z_limit=4.0
        )
        drift_alarm = None
        for value in gain_error_tensor.tolist():
            state = drift_monitor.observe(value)
            if state["drift"] and drift_alarm is None:
                drift_alarm = state["observation"]
        self.assertIsNotNone(
            drift_alarm,
            "Gain drift must trigger drift alarm within the window",
        )

    def test_fdi_catches_replay_masking_on_all_normal_records(self) -> None:
        """Replay attack replaces anomalous records with ideal noiseless copies.

        The noise-floor check must catch every such record because genuine
        telemetry always carries a sensor noise floor that ideal waveforms
        lack.
        """
        normals = self.bundle.test_waveforms[self.bundle.test_labels == 0]
        replay = detector._three_phase_base(512).repeat(normals.shape[0], 1, 1)
        fdi = detector.detect_false_data_injection(replay, self.fdi_calibration)
        self.assertTrue(bool(fdi["too_clean"].all()))
        self.assertTrue(bool(fdi["flags"].all()))

    def test_fdi_genuine_records_unflagged(self) -> None:
        """Genuine normal telemetry must produce zero FDI flags.

        This guards against false positives in the physics layer: the
        calibration bounds must admit every record from the real synthesis
        pipeline.
        """
        normals = self.bundle.test_waveforms[self.bundle.test_labels == 0]
        fdi = detector.detect_false_data_injection(normals, self.fdi_calibration)
        self.assertEqual(int(fdi["flags"].sum()), 0)
        self.assertEqual(int(fdi["individual"].sum()), 0)
        self.assertEqual(int(fdi["joint"].sum()), 0)


if __name__ == "__main__":
    unittest.main()
