import os
import random
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from cocotb_tools.runner import get_runner

def twos(val, bits):
    return val & ((1 << bits) - 1)

def as_signed(val, bits):
    val = twos(val, bits)
    if val & (1 << (bits - 1)):
        val -= (1 << bits)
    return val

def sat_to_bits(v, out_w):
    mn = -(1 << (out_w - 1))
    mx = (1 << (out_w - 1)) - 1
    if v > mx:
        return mx
    if v < mn:
        return mn
    return v

def pack_list_signed(vals, w):
    out = 0
    for i, v in enumerate(vals):
        out |= (twos(v, w) << (i * w))
    return out

def model_simple(x, w, bias, NUM_INPUTS=8, X_W=8, W_W=8, B_W=16, OUT_W=16, USE_RELU=1):
    acc = as_signed(bias, B_W)
    for i in range(NUM_INPUTS):
        acc += as_signed(x[i], X_W) * as_signed(w[i], W_W)
    if USE_RELU and acc < 0:
        acc = 0
    return sat_to_bits(acc, OUT_W)

def read_signed(handle):
    """Compatible with cocotb versions that expose signed_integer or LogicArray.to_signed()."""
    v = handle.value
    if hasattr(v, "signed_integer"):
        return int(v.signed_integer)
    if hasattr(v, "to_signed"):
        return int(v.to_signed())
    return int(v)

async def generate_clock(dut, period_ns=10):
    """Generate clock pulses (Timer-based, like the example)."""
    half = period_ns // 2
    while True:
        dut.clk.value = 0
        await Timer(half, unit="ns")
        dut.clk.value = 1
        await Timer(half, unit="ns")

async def reset_dut(dut, cycles=5):
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.bias.value = 0
    dut.x_flat.value = 0
    dut.w_flat.value = 0
    for _ in range(cycles):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

async def apply_and_check_one(dut, x, w, bias, NUM_INPUTS=8, X_W=8, W_W=8, B_W=16, OUT_W=16, USE_RELU=1,
                              max_wait_cycles=2000):
    """Drive one transaction and check result."""
    dut.x_flat.value = pack_list_signed(x, X_W)
    dut.w_flat.value = pack_list_signed(w, W_W)
    dut.bias.value   = bias

    # Wait for in_ready
    for _ in range(max_wait_cycles):
        if int(dut.in_ready.value) == 1:
            break
        await RisingEdge(dut.clk)
    else:
        raise AssertionError("Timeout waiting for in_ready==1")

    # Pulse in_valid for one cycle
    dut.in_valid.value = 1
    await RisingEdge(dut.clk)
    dut.in_valid.value = 0

    # Wait for out_valid
    for _ in range(max_wait_cycles):
        if int(dut.out_valid.value) == 1:
            break
        await RisingEdge(dut.clk)
    else:
        raise AssertionError("Timeout waiting for out_valid==1")

    got = read_signed(dut.out_data)
    exp = model_simple(x, w, bias, NUM_INPUTS=NUM_INPUTS, X_W=X_W, W_W=W_W, B_W=B_W, OUT_W=OUT_W, USE_RELU=USE_RELU)

    assert got == exp, f"Mismatch: got={got}, exp={exp}, x={x}, w={w}, bias={bias}"


# ----------------------------
# Tests (4 test cases)
# ----------------------------
@cocotb.test()
async def test_known_vector(dut):
    """Directed test: known small values (no saturation, no ReLU clamp)."""
    cocotb.start_soon(generate_clock(dut))
    await reset_dut(dut)

    NUM_INPUTS = 8
    X_W = 8
    W_W = 8
    B_W = 16
    OUT_W = 16
    USE_RELU = 1

    x = [1, 2, 3, 4, 5, 6, 7, 8]
    w = [1, 1, 1, 1, 1, 1, 1, 1]
    bias = 10  # sum(x)=36 => 46

    await apply_and_check_one(dut, x, w, bias, NUM_INPUTS, X_W, W_W, B_W, OUT_W, USE_RELU)


@cocotb.test()
async def test_relu_clamp_to_zero(dut):
    """Directed test: negative accumulation should clamp to 0 when USE_RELU=1."""
    cocotb.start_soon(generate_clock(dut))
    await reset_dut(dut)

    NUM_INPUTS = 8
    X_W = 8
    W_W = 8
    B_W = 16
    OUT_W = 16
    USE_RELU = 1

    x = [-1] * NUM_INPUTS
    w = [100] * NUM_INPUTS
    bias = 0  # acc = -800 => ReLU => 0

    await apply_and_check_one(dut, x, w, bias, NUM_INPUTS, X_W, W_W, B_W, OUT_W, USE_RELU)


@cocotb.test()
async def test_positive_saturation(dut):
    """Directed test: force positive overflow and confirm saturation to +32767 for OUT_W=16."""
    cocotb.start_soon(generate_clock(dut))
    await reset_dut(dut)

    NUM_INPUTS = 8
    X_W = 8
    W_W = 8
    B_W = 16
    OUT_W = 16
    USE_RELU = 1

    x = [127] * NUM_INPUTS
    w = [127] * NUM_INPUTS
    bias = 0  # huge positive => saturate

    await apply_and_check_one(dut, x, w, bias, NUM_INPUTS, X_W, W_W, B_W, OUT_W, USE_RELU)


@cocotb.test()
async def test_random_regression(dut):
    """Random regression: 50 randomized vectors with fixed seed."""
    cocotb.start_soon(generate_clock(dut))
    await reset_dut(dut)

    random.seed(2)

    NUM_INPUTS = 8
    X_W = 8
    W_W = 8
    B_W = 16
    OUT_W = 16
    USE_RELU = 1

    for _ in range(50):
        x = [random.randint(-(1 << (X_W - 1)), (1 << (X_W - 1)) - 1) for _ in range(NUM_INPUTS)]
        w = [random.randint(-(1 << (W_W - 1)), (1 << (W_W - 1)) - 1) for _ in range(NUM_INPUTS)]
        bias = random.randint(-(1 << (B_W - 1)), (1 << (B_W - 1)) - 1)

        await apply_and_check_one(dut, x, w, bias, NUM_INPUTS, X_W, W_W, B_W, OUT_W, USE_RELU)


def test_neuron_mac_simple_hidden_runner():
    sim = os.getenv("SIM", "icarus")
    proj_path = Path(__file__).resolve().parent.parent

    sources = [proj_path / "sources/neuron_mac_simple.v"]

    runner = get_runner(sim)
    runner.build(
        sources=sources,
        hdl_toplevel="neuron_mac_simple",
        always=True,
    )
    runner.test(
        hdl_toplevel="neuron_mac_simple",
        test_module="test_neuron_mac_simple_hidden",
    )
