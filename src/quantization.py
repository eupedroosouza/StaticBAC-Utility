import numpy as np


def convert_bitdepth(q_int32, bitwidth):
    qmin = -(1 << (bitwidth - 1))
    qmax = (1 << (bitwidth - 1)) - 1
    return np.clip(q_int32, qmin, qmax).astype(np.int32)


def optimal_uniform_quant(x, bitwidth, search_steps=40):
    x = x.astype(np.float32)
    Qmax = (1 << (bitwidth - 1)) - 1

    if x.size == 0 or np.all(x == 0):
        return np.zeros_like(x, dtype=np.int32), 1.0

    std = float(np.std(x))
    if std == 0:
        return np.zeros_like(x, dtype=np.int32), 1.0

    qstep_min = max(std / (1 << (bitwidth + 2)), 1e-12)
    qstep_max = max(std * 4.0, qstep_min * 2.0)

    phi = (1 + np.sqrt(5)) / 2.0
    invphi = 1.0 / phi

    a, b = qstep_min, qstep_max
    c = b - (b - a) * invphi
    d = a + (b - a) * invphi

    def mse(qstep):
        q = np.clip(np.round(x / qstep), -Qmax, Qmax)
        return np.mean((x - q * qstep) ** 2)

    fc, fd = mse(c), mse(d)

    for _ in range(search_steps):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - (b - a) * invphi
            fc = mse(c)
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) * invphi
            fd = mse(d)

    qstep = (a + b) / 2.0
    q = np.clip(np.round(x / qstep), -Qmax, Qmax)

    return q.astype(np.int32), float(qstep)


def quantize_tensor(arr: np.float32 | np.int32, use_quant: bool = True, tensor_kind: str = "weight"):
    numel = arr.size

    if tensor_kind == "buffer":
        return arr, 1.0, 32

    if not use_quant:
        # assume already quantized
        return arr, 1.0, 8

    if numel < 32:
        bitwidth = 12  # Change this if you want more precision (improve accurracy)
        qstep = np.max(np.abs(arr)) / (2 ** (bitwidth - 1) - 1 + 1e-8)
        q = np.round(arr / qstep)

    elif tensor_kind == "weight":
        bitwidth = 8
        q, qstep = optimal_uniform_quant(arr, bitwidth)
    else:
        bitwidth = 12  # Change this if you want more precision (improve accurracy)
        q, qstep = optimal_uniform_quant(arr, bitwidth)

    q = convert_bitdepth(q, bitwidth)

    return q.astype(np.int32), qstep, bitwidth
