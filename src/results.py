import csv
import re
from pathlib import Path

import app
import inference
import sbac


def extract(pattern: str, text: str, flags=0):
    match = re.search(pattern, text, flags)
    return match.group(1) if match else "N/A"


def to_num(val):
    if val == "N/A":
        return val
    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            return val


def save_result(path: Path, ctx: app.Context, result: app.IterationResult):
    fields = ["model", "tensors", "original_size", "compressed_size", "compression_ratio", "encoding_time",
              "encoding_speed", "encoding_mem_baseline", "encoding_mem_peak",
              "encoding_mem_delta", "decoding_time", "decoding_speed", "decoding_mem_baseline", "decoding_mem_peak",
              "decoding_mem_delta", "overall_max_error", "overall_mean_error", "dataset", "accuracy_metric"]

    row = [ctx.model.name, result.staticbac.tensors, f"{result.staticbac.encode.originalSize:.2f}",
           f"{result.staticbac.encode.compressedSize:.2f}",
           f"{result.staticbac.encode.compressionRatio:.4f}", f"{result.staticbac.encode.time:.4f}",
           f"{result.staticbac.encode.speed:.4f}", f"{result.staticbac.encode.mem.baseline:.4f}",
           f"{result.staticbac.encode.mem.peak:.4f}", f"{result.staticbac.encode.mem.delta:.4f}",
           f"{result.staticbac.decode.time:.4f}", f"{result.staticbac.decode.speed:.4f}",
           f"{result.staticbac.decode.mem.baseline:.4f}", f"{result.staticbac.decode.mem.peak:.4f}",
           f"{result.staticbac.decode.mem.delta:.4f}", f"{result.overall_max_error:.20f}",
           f"{result.overall_mean_error:.20f}", inference.info[ctx.model.inference].name, result.accuracy_metric]

    for k, v in result.accuracy_result.items():
        fields.append(f"accuracy_{k}")
        row.append(f"{v:.4f}")

    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        writer.writerow(row)
    pass


def avg_sbac_results(results: list[sbac.StaticBacExecResult]) -> sbac.StaticBacExecResult | None:
    len_r = len(results)
    if len_r == 0:
        return None
    if len_r == 1:
        return results[0]

    base_res = results[len_r - 1]

    result = sbac.StaticBacExecResult(
        base_res.tensors,
        sbac.StaticBacEncodeResult(
            base_res.encode.originalSize,
            base_res.encode.compressedSize,
            base_res.encode.compressionRatio,
            sum(r.encode.time for r in results) / len_r,
            sum(r.encode.speed for r in results) / len_r,
            sbac.StaticBacMemResult(
                sum(r.encode.mem.baseline for r in results) / len_r,
                sum(r.encode.mem.peak for r in results) / len_r,
                sum(r.encode.mem.delta for r in results) / len_r
            )
        ),
        sbac.StaticBacDecodeResult(
            base_res.decode.decodedSize,
            sum(r.decode.time for r in results) / len_r,
            sum(r.decode.speed for r in results) / len_r,
            sbac.StaticBacMemResult(
                sum(r.decode.mem.baseline for r in results) / len_r,
                sum(r.decode.mem.peak for r in results) / len_r,
                sum(r.decode.mem.delta for r in results) / len_r
            )
        ),
        base_res.result  # Dummy
    )

    return result
