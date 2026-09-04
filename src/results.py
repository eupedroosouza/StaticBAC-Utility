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
    data: dict[str, str] = {}

    data["model"] = ctx.model.name
    data["tensors"] = f"{result.staticbac.tensors}"
    data["original_size"] = f"{result.staticbac.encode.originalSize:.2f}"
    data["compressed_size"] = f"{result.staticbac.encode.compressedSize:.2f}"
    data["decoded_size"] = f"{result.staticbac.decode.decodedSize:.2f}"
    data["compression_ratio"] = f"{result.staticbac.encode.compressionRatio:.4f}"
    data["encode_time"] = f"{result.staticbac.encode.time:.4f}"
    data["encode_speed"] = f"{result.staticbac.encode.speed:.4f}"
    data["decode_time"] = f"{result.staticbac.decode.time:.4f}"
    data["decode_speed"] = f"{result.staticbac.decode.speed:.4f}"
    data["encode_mem_baseline"] = f"{result.staticbac.encode.mem.baseline:.4f}"
    data["encode_mem_peak"] = f"{result.staticbac.encode.mem.peak:.4f}"
    data["encode_mem_delta"] = f"{result.staticbac.encode.mem.delta:.4f}"
    data["decode_mem_baseline"] = f"{result.staticbac.decode.mem.baseline:.4f}"
    data["decode_mem_peak"] = f"{result.staticbac.decode.mem.peak:.4f}"
    data["decode_mem_delta"] = f"{result.staticbac.decode.mem.delta:.4f}"
    data["dataset"] = inference.info[ctx.model.inference].name
    data["acuracy_metric"] = result.accuracy_metric


    for k, v in result.accuracy_result.items():
        data[f"accuracy_{k}"] = f"{v:.4f}"

    data["overall_max_error"] = f"{result.overall_max_error:.20f}"
    data["overall_mean_error"] = f"{result.overall_mean_error:.20f}"

    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        for name, value in data.items():
            writer.writerow([name, value])
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
