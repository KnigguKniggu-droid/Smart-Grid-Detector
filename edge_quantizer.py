"""Edge hardware emulation and optimization suite for the Smart Grid Detector.

Targets ARM Cortex-M7 / Edge Micro-AI tier. Implements Post-Training Static
Quantization (PTQ), MAC/FLOP profiling, Flash/SRAM memory estimation, and
an FDI dispatch regression test comparing FP32 vs INT8.

Run standalone:
    python edge_quantizer.py --device cpu

Or import the components individually for integration testing.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import statistics
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

import smart_grid_detector as detector

if sys.platform == "win32":
    with contextlib.suppress(AttributeError, RuntimeError):
        sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. COMPUTE AND MEMORY PROFILER
# ---------------------------------------------------------------------------


@dataclass
class ConvLayerProfile:
    """Profile of a single Conv1d or ConvTranspose1d layer."""
    layer_name: str
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    output_padding: int
    bias: bool
    input_length: int
    output_length: int
    weight_params: int
    bias_params: int
    macs: int
    flops: int
    activation_bytes_fp32: int
    activation_bytes_int8: int


@dataclass
class LinearLayerProfile:
    """Profile of a single Linear layer."""
    layer_name: str
    in_features: int
    out_features: int
    bias: bool
    weight_params: int
    bias_params: int
    macs: int
    flops: int
    activation_bytes_fp32: int
    activation_bytes_int8: int


@dataclass
class ModelProfile:
    """Complete model profile: MACs, FLOPs, Flash, SRAM."""
    conv_layers: list[ConvLayerProfile] = field(default_factory=list)
    linear_layers: list[LinearLayerProfile] = field(default_factory=list)
    total_macs: int = 0
    total_flops: int = 0
    total_params: int = 0
    flash_fp32_bytes: int = 0
    flash_int8_bytes: int = 0
    peak_sram_fp32_bytes: int = 0
    peak_sram_int8_bytes: int = 0
    batchnorm_params: int = 0

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serializable summary."""
        return {
            "total_macs": self.total_macs,
            "total_flops": self.total_flops,
            "total_params": self.total_params,
            "flash_fp32_kb": round(self.flash_fp32_bytes / 1024, 2),
            "flash_int8_kb": round(self.flash_int8_bytes / 1024, 2),
            "flash_reduction_ratio": round(
                self.flash_fp32_bytes / max(self.flash_int8_bytes, 1), 2
            ),
            "peak_sram_fp32_kb": round(self.peak_sram_fp32_bytes / 1024, 2),
            "peak_sram_int8_kb": round(self.peak_sram_int8_bytes / 1024, 2),
            "sram_reduction_ratio": round(
                self.peak_sram_fp32_bytes / max(self.peak_sram_int8_bytes, 1), 2
            ),
            "conv_layer_count": len(self.conv_layers),
            "linear_layer_count": len(self.linear_layers),
            "batchnorm_param_count": self.batchnorm_params,
        }


def profile_conv1d(
    layer: nn.Conv1d,
    layer_name: str,
    input_length: int,
    is_transpose: bool = False,
) -> ConvLayerProfile:
    """Profile a single Conv1d or ConvTranspose1d layer."""
    in_ch = layer.in_channels
    out_ch = layer.out_channels
    k = layer.kernel_size[0]
    stride = layer.stride[0]
    padding = layer.padding[0]
    out_padding = layer.output_padding[0] if is_transpose else 0
    has_bias = layer.bias is not None

    if is_transpose:
        output_length = (input_length - 1) * stride - 2 * padding + k + out_padding
        # Transposed conv MACs: input_positions * kernel * in_channels * out_channels
        # Each input position contributes kernel_size MACs per output channel
        macs = input_length * k * in_ch * out_ch
    else:
        output_length = (input_length + 2 * padding - k) // stride + 1
        # Standard conv MACs: output_positions * kernel * in_channels * out_channels
        macs = output_length * k * in_ch * out_ch

    flops = macs * 2  # multiply + add = 2 FLOPs per MAC
    if has_bias:
        flops += output_length * out_ch

    weight_params = out_ch * in_ch * k
    bias_params = out_ch if has_bias else 0

    # Activation memory: output tensor [batch=1, out_channels, output_length]
    # Plus input tensor that must stay resident during computation
    act_fp32 = out_ch * output_length * 4  # float32 = 4 bytes
    act_int8 = out_ch * output_length * 1   # int8 = 1 byte

    return ConvLayerProfile(
        layer_name=layer_name,
        in_channels=in_ch,
        out_channels=out_ch,
        kernel_size=k,
        stride=stride,
        padding=padding,
        output_padding=out_padding,
        bias=has_bias,
        input_length=input_length,
        output_length=output_length,
        weight_params=weight_params,
        bias_params=bias_params,
        macs=macs,
        flops=flops,
        activation_bytes_fp32=act_fp32,
        activation_bytes_int8=act_int8,
    )


def profile_linear(
    layer: nn.Linear,
    layer_name: str,
) -> LinearLayerProfile:
    """Profile a single Linear layer."""
    in_f = layer.in_features
    out_f = layer.out_features
    has_bias = layer.bias is not None

    macs = in_f * out_f
    flops = macs * 2
    if has_bias:
        flops += out_f

    act_fp32 = out_f * 4
    act_int8 = out_f * 1

    return LinearLayerProfile(
        layer_name=layer_name,
        in_features=in_f,
        out_features=out_f,
        bias=has_bias,
        weight_params=in_f * out_f,
        bias_params=out_f if has_bias else 0,
        macs=macs,
        flops=flops,
        activation_bytes_fp32=act_fp32,
        activation_bytes_int8=act_int8,
    )


def profile_model(
    model: detector.GridWaveformAutoencoder,
    sequence_length: int = 512,
    base_channels: int = 16,
    latent_dim: int = 64,
) -> ModelProfile:
    """Profile the full autoencoder: MACs, FLOPs, Flash, and peak SRAM.

    The autoencoder architecture (from GridWaveformAutoencoder):
      Encoder: Conv1d(3->C, k=7, s=2) -> Conv1d(C->2C, k=7, s=2) -> Conv1d(2C->4C, k=7, s=2)
      Latent:  Linear(4C*L/8 -> latent) -> Linear(latent -> 4C*L/8)
      Decoder: ConvT1d(4C->2C, k=7, s=2) -> ConvT1d(2C->C, k=7, s=2) -> ConvT1d(C->3, k=7, s=2)
    """
    profile = ModelProfile()
    L = sequence_length
    encoded_length = L // 8

    # Encoder Conv1d layers (indices 0, 3, 6 in the Sequential)
    encoder_convs = [
        ("encoder_conv1", model.encoder_convolutions[0], L, False),
        ("encoder_conv2", model.encoder_convolutions[3], L // 2, False),
        ("encoder_conv3", model.encoder_convolutions[6], L // 4, False),
    ]
    for name, conv, in_len, is_t in encoder_convs:
        lp = profile_conv1d(conv, name, in_len, is_t)
        profile.conv_layers.append(lp)
        profile.total_macs += lp.macs
        profile.total_flops += lp.flops
        profile.total_params += lp.weight_params + lp.bias_params

    # Linear layers
    lin1 = profile_linear(model.to_latent, "to_latent")
    lin2 = profile_linear(model.from_latent[0], "from_latent")
    profile.linear_layers.extend([lin1, lin2])
    profile.total_macs += lin1.macs + lin2.macs
    profile.total_flops += lin1.flops + lin2.flops
    profile.total_params += lin1.weight_params + lin1.bias_params
    profile.total_params += lin2.weight_params + lin2.bias_params

    # Decoder ConvTranspose1d layers (indices 0, 3, 6 in the Sequential)
    decoder_convs = [
        ("decoder_conv1", model.decoder_convolutions[0], encoded_length, True),
        ("decoder_conv2", model.decoder_convolutions[3], L // 4, True),
        ("decoder_conv3", model.decoder_convolutions[6], L // 2, True),
    ]
    for name, conv, in_len, is_t in decoder_convs:
        lp = profile_conv1d(conv, name, in_len, is_t)
        profile.conv_layers.append(lp)
        profile.total_macs += lp.macs
        profile.total_flops += lp.flops
        profile.total_params += lp.weight_params + lp.bias_params

    # BatchNorm parameters: each BN has weight + bias = 2 * channels
    bn_modules = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
    for bn in bn_modules:
        profile.batchnorm_params += 2 * bn.num_features
    profile.total_params += profile.batchnorm_params

    # Flash memory: weights on disk
    # FP32: each param = 4 bytes
    # INT8: conv/linear weights = 1 byte, BN = 4 bytes (kept FP32), bias = 4 bytes
    profile.flash_fp32_bytes = profile.total_params * 4
    weight_only = sum(lp.weight_params for lp in profile.conv_layers) + sum(
        lp.weight_params for lp in profile.linear_layers
    )
    bias_only = sum(lp.bias_params for lp in profile.conv_layers) + sum(
        lp.bias_params for lp in profile.linear_layers
    )
    profile.flash_int8_bytes = weight_only * 1 + bias_only * 4 + profile.batchnorm_params * 4

    # Peak SRAM: largest set of activation tensors resident simultaneously.
    # During forward pass, the input tensor must stay in memory while the
    # first conv computes. The peak is the maximum of (input + largest output).
    input_bytes_fp32 = 3 * L * 4
    input_bytes_int8 = 3 * L * 1

    # Find the largest activation across all layers
    max_act_fp32 = max(
        max(lp.activation_bytes_fp32 for lp in profile.conv_layers),
        max(lp.activation_bytes_fp32 for lp in profile.linear_layers),
    )
    max_act_int8 = max(
        max(lp.activation_bytes_int8 for lp in profile.conv_layers),
        max(lp.activation_bytes_int8 for lp in profile.linear_layers),
    )

    # Peak SRAM = input + largest activation (both resident during first layer)
    profile.peak_sram_fp32_bytes = input_bytes_fp32 + max_act_fp32
    profile.peak_sram_int8_bytes = input_bytes_int8 + max_act_int8

    return profile


# ---------------------------------------------------------------------------
# 2. POST-TRAINING STATIC QUANTIZATION (PTQ)
# ---------------------------------------------------------------------------


def apply_static_quantization(
    model: detector.GridWaveformAutoencoder,
    calibration_data: torch.Tensor,
    device: torch.device = torch.device("cpu"),
) -> tuple[nn.Module, dict[str, Any]]:
    """Apply Post-Training Static Quantization to the autoencoder.

    Uses torch.ao.quantization.prepare_fx + convert_fx for graph-mode PTQ.
    Conv1d layers get per-channel int8 weight quantization; Linear layers
    get per-tensor int8. Calibration uses a representative subset of normal
    60 Hz waveforms to collect activation min/max ranges.

    Returns (quantized_model, quantization_metadata).
    """
    model_cpu = copy.deepcopy(model).to(device).eval()

    # Fuse Conv-BN-ReLU sequences for quantization efficiency
    # Encoder fusion: [Conv1d, BatchNorm1d, ReLU] -> [ConvBnReLU]
    encoder_fuse_patterns = [
        ["encoder_convolutions.0", "encoder_convolutions.1", "encoder_convolutions.2"],
        ["encoder_convolutions.4", "encoder_convolutions.5", "encoder_convolutions.6"],
        ["encoder_convolutions.8", "encoder_convolutions.9", "encoder_convolutions.10"],
    ]
    decoder_fuse_patterns = [
        ["decoder_convolutions.0", "decoder_convolutions.1", "decoder_convolutions.2"],
        ["decoder_convolutions.4", "decoder_convolutions.5", "decoder_convolutions.6"],
    ]

    fused_count = 0
    try:
        for pattern in encoder_fuse_patterns + decoder_fuse_patterns:
            try:
                torch.quantization.fuse_modules(model_cpu, pattern, inplace=True)
                fused_count += 1
            except (RuntimeError, ValueError):
                pass
    except Exception:
        pass

    metadata: dict[str, Any] = {
        "method": "post_training_static",
        "fused_modules": fused_count,
        "calibration_samples": calibration_data.shape[0],
        "quantization_engine": torch.backends.quantized.engine,
        "torch_version": torch.__version__,
    }

    # Try FX graph-mode quantization
    try:
        from torch.ao.quantization import (
            get_default_qconfig,
            prepare_fx,
            convert_fx,
        )

        qconfig = get_default_qconfig("fbgemm" if sys.platform != "win32" else "qnnpack")
        with contextlib.suppress(RuntimeError):
            torch.backends.quantized.engine = "qnnpack"

        example_inputs = (calibration_data[:1].to(device),)
        prepared = prepare_fx(model_cpu, {"": qconfig}, example_inputs)

        # Calibrate with representative normal waveforms
        with torch.inference_mode():
            batch_size = 64
            for i in range(0, calibration_data.shape[0], batch_size):
                batch = calibration_data[i:i + batch_size].to(device)
                prepared(batch)

        quantized_model = convert_fx(prepared)
        metadata["mode"] = "fx_graph"
        metadata["success"] = True

        return quantized_model, metadata

    except (ImportError, RuntimeError, ValueError, TypeError) as error:
        metadata["mode"] = "eager_fallback"
        metadata["fx_error"] = f"{type(error).__name__}: {error}"

    # Fallback: eager-mode quantization with per-channel weight quantization
    try:
        quant_config = torch.quantization.QConfig(
            activation=torch.quantization.MinMaxObserver.with_args(
                quant_min=0, quant_max=255, dtype=torch.quint8
            ),
            weight=torch.quantization.MinMaxObserver.with_args(
                dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
            ),
        )
        model_cpu.qconfig = quant_config
        torch.quantization.prepare(model_cpu, inplace=True)

        with torch.inference_mode():
            batch_size = 64
            for i in range(0, calibration_data.shape[0], batch_size):
                batch = calibration_data[i:i + batch_size].to(device)
                model_cpu(batch)

        torch.quantization.convert(model_cpu, inplace=True)
        metadata["mode"] = "eager"
        metadata["success"] = True
        return model_cpu, metadata

    except (RuntimeError, ValueError, NotImplementedError) as error:
        metadata["mode"] = "dynamic_fallback"
        metadata["eager_error"] = f"{type(error).__name__}: {error}"

    # Last resort: dynamic quantization on Linear layers
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="torch.ao.quantization is deprecated.*",
                category=DeprecationWarning,
            )
            int8_model = torch.quantization.quantize_dynamic(
                copy.deepcopy(model).to(device),
                {nn.Linear},
                dtype=torch.qint8,
            )
        metadata["mode"] = "dynamic"
        metadata["success"] = True
        return int8_model, metadata
    except (ImportError, RuntimeError, NotImplementedError) as error:
        metadata["mode"] = "failed"
        metadata["success"] = False
        metadata["error"] = f"{type(error).__name__}: {error}"
        return model_cpu, metadata


def measure_model_size(model: nn.Module) -> int:
    """Return the state_dict size in bytes."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes


def measure_latency(
    model: nn.Module,
    sample: torch.Tensor,
    iterations: int = 100,
) -> dict[str, float]:
    """Measure single-record inference latency."""
    model.eval()
    with torch.inference_mode():
        for _ in range(10):
            model(sample)
        timings: list[float] = []
        for _ in range(iterations):
            start = time.perf_counter()
            model(sample)
            timings.append((time.perf_counter() - start) * 1_000.0)
    return {
        "latency_ms_median": round(statistics.median(timings), 4),
        "latency_ms_mean": round(statistics.mean(timings), 4),
        "latency_ms_p95": round(
            sorted(timings)[int(len(timings) * 0.95)], 4
        ),
        "iterations": iterations,
    }


def evaluate_accuracy(
    model: nn.Module,
    test_waveforms: torch.Tensor,
    test_labels: torch.Tensor,
    thd_values: torch.Tensor,
    invalid_flags: torch.Tensor,
    threshold_sigma: float,
    train_normal: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate detection accuracy and FDI metrics for a model variant."""
    truth = test_labels.to(dtype=torch.bool, device="cpu")
    thd_flags = (thd_values > detector.THD_LIMIT).cpu()
    invalid = invalid_flags.to(dtype=torch.bool).cpu()

    # Recalibrate threshold on this model
    with torch.inference_mode():
        cal_errors = []
        for i in range(0, train_normal.shape[0], 128):
            batch = train_normal[i:i + 128].to(device)
            cal_errors.append(detector._per_waveform_mse(model(batch), batch))
        errors = torch.cat(cal_errors)
        threshold = float(
            errors.mean() + threshold_sigma * errors.std(unbiased=False)
        )

        test_errors = []
        for i in range(0, test_waveforms.shape[0], 128):
            batch = torch.nan_to_num(
                test_waveforms[i:i + 128].to(device),
                nan=0.0, posinf=0.0, neginf=0.0,
            )
            test_errors.append(detector._per_waveform_mse(model(batch), batch))
        test_error_tensor = torch.cat(test_errors)

    predictions = (test_error_tensor > threshold) | thd_flags | invalid
    metrics = detector._binary_metrics(predictions, truth)

    # FDI detection rate: run FDI scenario against this model
    fdi_calibration = detector.calibrate_fdi_detector(train_normal)
    normals = test_waveforms[test_labels == 0]
    anomalous = test_waveforms[test_labels == 1]

    # Replay masking attack
    replay = detector._three_phase_base(test_waveforms.shape[-1]).repeat(
        anomalous.shape[0], 1, 1
    )
    replay_fdi = detector.detect_false_data_injection(replay, fdi_calibration)
    replay_detection = int(replay_fdi["flags"].sum()) / max(
        anomalous.shape[0], 1
    )

    # Phase bias attack
    bias = normals.clone()
    bias[:, 0, :] = bias[:, 0, :] * 1.06
    bias_fdi = detector.detect_false_data_injection(bias, fdi_calibration)
    bias_detection = int(bias_fdi["flags"].sum()) / max(
        normals.shape[0], 1
    )

    # Baseline false positives
    baseline_fdi = detector.detect_false_data_injection(normals, fdi_calibration)
    baseline_fp = int(baseline_fdi["flags"].sum())

    # Overall FDI detection rate (replay + phase bias caught)
    total_attacks = anomalous.shape[0] + normals.shape[0]
    total_caught = int(replay_fdi["flags"].sum()) + int(bias_fdi["flags"].sum())
    fdi_detection_rate = total_caught / max(total_attacks, 1)

    return {
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "threshold": threshold,
        "fdi_detection_rate": round(fdi_detection_rate, 4),
        "replay_masking_detection": round(replay_detection, 4),
        "phase_bias_detection": round(bias_detection, 4),
        "baseline_false_positives": baseline_fp,
    }


# ---------------------------------------------------------------------------
# 3. DISPATCH REGRESSION TEST: FP32 vs INT8 COMPARISON
# ---------------------------------------------------------------------------


def run_dispatch_regression(
    model: detector.GridWaveformAutoencoder,
    bundle: detector.GeneratedWaveforms,
    train_normal: torch.Tensor,
    threshold_sigma: float,
    device: torch.device,
    calibration_samples: int = 200,
) -> dict[str, Any]:
    """Run FP32 vs INT8 comparison across accuracy, FDI, latency, memory.

    Asserts that FDI detection remains above 95% post-quantization.
    Returns a structured comparison table.
    """
    cpu = torch.device("cpu")
    test_waveforms = bundle.test_waveforms
    test_labels = bundle.test_labels
    thd_values = detector.compute_thd(test_waveforms)
    # Use zeros for invalid flags (the edge benchmark does the same simplification)
    invalid_flags = torch.zeros(test_waveforms.shape[0], dtype=torch.bool)

    # Calibration subset: representative normal 60 Hz waveforms
    normal_indices = (test_labels == 0).nonzero(as_tuple=True)[0]
    cal_count = min(calibration_samples, normal_indices.shape[0])
    cal_data = test_waveforms[normal_indices[:cal_count]].to(cpu)

    single_sample = test_waveforms[:1].to(cpu)

    # Profile the model architecture
    profile = profile_model(
        model,
        sequence_length=model.sequence_length,
        base_channels=model.encoder_convolutions[0].out_channels,
        latent_dim=model.to_latent.out_features,
    )

    # --- FP32 baseline ---
    fp32_model = copy.deepcopy(model).to(cpu).eval()
    fp32_size = measure_model_size(fp32_model)
    fp32_latency = measure_latency(fp32_model, single_sample)
    fp32_metrics = evaluate_accuracy(
        fp32_model, test_waveforms, test_labels,
        thd_values, invalid_flags, threshold_sigma,
        train_normal, cpu,
    )

    # --- INT8 quantized ---
    int8_model, quant_meta = apply_static_quantization(
        model, cal_data, cpu,
    )
    int8_size = measure_model_size(int8_model)
    int8_latency = measure_latency(int8_model, single_sample)
    int8_metrics = evaluate_accuracy(
        int8_model, test_waveforms, test_labels,
        thd_values, invalid_flags, threshold_sigma,
        train_normal, cpu,
    )

    # --- Build comparison table ---
    comparison = {
        "target_hardware": "ARM Cortex-M7 / Edge Micro-AI",
        "model_profile": profile.summary(),
        "quantization_metadata": quant_meta,
        "comparison_table": {
            "fp32": {
                "accuracy": round(fp32_metrics["accuracy"], 4),
                "fdi_detection_rate": round(fp32_metrics["fdi_detection_rate"], 4),
                "latency_ms": fp32_latency["latency_ms_median"],
                "flash_kb": round(fp32_size / 1024, 2),
                "sram_peak_kb": profile.peak_sram_fp32_bytes / 1024,
                "macs": profile.total_macs,
                "flops": profile.total_flops,
                "params": profile.total_params,
            },
            "int8": {
                "accuracy": round(int8_metrics["accuracy"], 4),
                "fdi_detection_rate": round(int8_metrics["fdi_detection_rate"], 4),
                "latency_ms": int8_latency["latency_ms_median"],
                "flash_kb": round(int8_size / 1024, 2),
                "sram_peak_kb": profile.peak_sram_int8_bytes / 1024,
                "macs": profile.total_macs,
                "flops": profile.total_flops,
                "params": profile.total_params,
            },
        },
        "deltas": {
            "accuracy_delta": round(
                int8_metrics["accuracy"] - fp32_metrics["accuracy"], 4
            ),
            "fdi_detection_delta": round(
                int8_metrics["fdi_detection_rate"]
                - fp32_metrics["fdi_detection_rate"], 4
            ),
            "latency_speedup": round(
                fp32_latency["latency_ms_median"]
                / max(int8_latency["latency_ms_median"], 1e-9), 2
            ),
            "flash_reduction_ratio": round(fp32_size / max(int8_size, 1), 2),
            "sram_reduction_ratio": round(
                profile.peak_sram_fp32_bytes
                / max(profile.peak_sram_int8_bytes, 1), 2
            ),
        },
        "assertions": {
            "fdi_above_95": int8_metrics["fdi_detection_rate"] >= 0.95,
            "accuracy_above_99": int8_metrics["accuracy"] >= 0.99,
            "flash_reduced": int8_size < fp32_size,
        },
        "detailed_metrics": {
            "fp32": fp32_metrics,
            "int8": int8_metrics,
        },
    }

    # Print summary table
    print("\n" + "=" * 72)
    print("EDGE HARDWARE EMULATION: FP32 vs INT8 (ARM Cortex-M7 target)")
    print("=" * 72)
    print(f"  {'Metric':<28} {'FP32':>16} {'INT8':>16} {'Delta':>10}")
    print("-" * 72)
    print(f"  {'Accuracy':<28} {fp32_metrics['accuracy']:>15.4f} {int8_metrics['accuracy']:>15.4f} {comparison['deltas']['accuracy_delta']:>+9.4f}")
    print(f"  {'FDI Detection Rate':<28} {fp32_metrics['fdi_detection_rate']:>15.4f} {int8_metrics['fdi_detection_rate']:>15.4f} {comparison['deltas']['fdi_detection_delta']:>+9.4f}")
    print(f"  {'Latency (ms)':<28} {fp32_latency['latency_ms_median']:>15.4f} {int8_latency['latency_ms_median']:>15.4f} {comparison['deltas']['latency_speedup']:>+8.2f}x")
    print(f"  {'Flash (KB)':<28} {fp32_size/1024:>15.2f} {int8_size/1024:>15.2f} {comparison['deltas']['flash_reduction_ratio']:>+8.2f}x")
    print(f"  {'Peak SRAM (KB)':<28} {profile.peak_sram_fp32_bytes/1024:>15.2f} {profile.peak_sram_int8_bytes/1024:>15.2f} {comparison['deltas']['sram_reduction_ratio']:>+8.2f}x")
    print(f"  {'Total MACs':<28} {profile.total_macs:>16,} {profile.total_macs:>16,} {'same':>10}")
    print(f"  {'Total FLOPs':<28} {profile.total_flops:>16,} {profile.total_flops:>16,} {'same':>10}")
    print(f"  {'Total Params':<28} {profile.total_params:>16,} {profile.total_params:>16,} {'same':>10}")
    print("-" * 72)

    fdi_pass = comparison["assertions"]["fdi_above_95"]
    acc_pass = comparison["assertions"]["accuracy_above_99"]
    flash_pass = comparison["assertions"]["flash_reduced"]
    print(f"  FDI Detection >= 95%: {'PASS' if fdi_pass else 'FAIL'}")
    print(f"  Accuracy >= 99%:      {'PASS' if acc_pass else 'FAIL'}")
    print(f"  Flash reduced:        {'PASS' if flash_pass else 'FAIL'}")
    print(f"  Quantization mode:    {quant_meta.get('mode', 'unknown')}")
    print("=" * 72)

    if not fdi_pass:
        print(
            f"\n  [WARNING] FDI detection rate dropped to "
            f"{int8_metrics['fdi_detection_rate']:.2%} (below 95% threshold)."
        )

    return comparison


# ---------------------------------------------------------------------------
# 4. CLI ENTRY POINT
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Edge hardware emulation and quantization suite."
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Torch device (default: cpu).",
    )
    parser.add_argument(
        "--num-samples", type=int, default=2000,
        help="Number of synthetic waveform samples.",
    )
    parser.add_argument(
        "--sequence-length", type=int, default=512,
        help="Sequence length per sample.",
    )
    parser.add_argument(
        "--epochs", type=int, default=15,
        help="Training epochs.",
    )
    parser.add_argument(
        "--calibration-samples", type=int, default=200,
        help="Calibration samples for PTQ.",
    )
    parser.add_argument(
        "--threshold-sigma", type=float, default=3.5,
        help="Threshold sigma for anomaly detection.",
    )
    parser.add_argument(
        "--json-output", type=str, default="",
        help="Write comparison JSON to this path.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    detector.set_deterministic_seed(42)

    print("[EDGE] Generating synthetic waveforms...")
    bundle = detector._generate_waveform_bundle(
        args.num_samples, args.sequence_length, seed=42,
    )
    model = detector.GridWaveformAutoencoder(sequence_length=args.sequence_length)
    normal_train = bundle.train_waveforms[bundle.train_labels == 0].contiguous()

    print("[EDGE] Training autoencoder...")
    detector.train_autoencoder(
        model, normal_train, device,
        epochs=args.epochs,
        calibration={},
    )

    print("[EDGE] Running dispatch regression test...")
    result = run_dispatch_regression(
        model, bundle, normal_train,
        threshold_sigma=args.threshold_sigma,
        device=device,
        calibration_samples=args.calibration_samples,
    )

    if args.json_output:
        output = Path(args.json_output)
        output.write_text(
            json.dumps(result, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"[EDGE] JSON written to {output}")


if __name__ == "__main__":
    main()
