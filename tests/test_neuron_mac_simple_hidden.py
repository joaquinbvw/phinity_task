import os
import random
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
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


def clog2_py(value):
    v = value - 1
    c = 0
    while v > 0:
        c += 1
        v >>= 1
    return c if c > 0 else 1


def wrap_signed(v, bits):
    return as_signed(twos(v, bits), bits)


def fx_from_float(r, frac):
    # Inputs are already quantized by the testbench; keep it simple.
    # Prefer using values that are exact multiples of 2^-frac to avoid ambiguity.
    return int(round(r * (1 << frac)))


def fx_to_float(v, frac):
    return float(v) / float(1 << frac)


def model_serial(
    x, w, bias,
    NUM_INPUTS=8,
    X_W=8, W_W=8, B_W=32, OUT_W=16,
    X_FRAC=4, W_FRAC=4, B_FRAC=8, OUT_FRAC=8,
    GUARD_BITS=2,
    USE_RELU=1
):
    """
    Bit-accurate model of neuron_mac_serial.v including:
      - bias alignment into FRAC_P = X_FRAC + W_FRAC (with rounding on right shifts)
      - serial accumulation with ACC_W wrap behavior
      - optional ReLU
      - output quantize to OUT_FRAC (with rounding on right shifts)
      - saturation to OUT_W
    """

    PROD_W = X_W + W_W
    FRAC_P = X_FRAC + W_FRAC
    SUM_GROW = 1 if NUM_INPUTS <= 1 else clog2_py(NUM_INPUTS)
    ACC_W = PROD_W + SUM_GROW + GUARD_BITS

    def round_shift_right_acc(v_acc, sh):
        # Matches RTL:
        #   round_const = 1 << (sh-1)
        #   if v >= 0: v += round_const else v -= round_const
        #   v = v >>> sh
        v_acc = wrap_signed(v_acc, ACC_W)
        if sh <= 0:
            return v_acc
        round_const = 1 << (sh - 1)
        if v_acc >= 0:
            v_acc = wrap_signed(v_acc + round_const, ACC_W)
        else:
            v_acc = wrap_signed(v_acc - round_const, ACC_W)
        # Python >> is arithmetic for negative ints (matches >>> on signed)
        v_acc = wrap_signed(v_acc >> sh, ACC_W)
        return v_acc

    def align_bias(b):
        # RTL does: reg signed [ACC_W-1:0] be; be = b;  (may truncate if B_W > ACC_W)
        b_bits = twos(b, B_W)
        be_bits = b_bits & ((1 << ACC_W) - 1)
        be = as_signed(be_bits, ACC_W)

        if B_FRAC > FRAC_P:
            sh = B_FRAC - FRAC_P
            be = round_shift_right_acc(be, sh)
        elif FRAC_P > B_FRAC:
            sh = FRAC_P - B_FRAC
            be = wrap_signed(be << sh, ACC_W)

        return wrap_signed(be, ACC_W)

    def sat_to_out(v_acc):
        # Compare against OUT_W signed limits, in current LSB units
        mx = (1 << (OUT_W - 1)) - 1
        mn = -(1 << (OUT_W - 1))
        if v_acc > mx:
            return mx
        if v_acc < mn:
            return mn
        # Match v[OUT_W-1:0] behavior
        return as_signed(twos(v_acc, OUT_W), OUT_W)

    def quantize_and_sat(vin_acc):
        v = wrap_signed(vin_acc, ACC_W)

        if USE_RELU:
            if v < 0:
                v = 0

        if FRAC_P > OUT_FRAC:
            sh = FRAC_P - OUT_FRAC
            v = round_shift_right_acc(v, sh)
        elif OUT_FRAC > FRAC_P:
            sh = OUT_FRAC - FRAC_P
            v = wrap_signed(v << sh, ACC_W)

        return sat_to_out(v)

    # Accumulate
    acc = align_bias(bias)
    for i in range(NUM_INPUTS):
        xi = as_signed(x[i], X_W)
        wi = as_signed(w[i], W_W)

        prod = xi * wi
        prod_s = as_signed(prod, PROD_W)   # matches signed [PROD_W-1:0] prod
        # prod_acc is sign-extended to ACC_W in RTL; integer value is the same
        acc = wrap_signed(acc + prod_s, ACC_W)

    # RTL outputs quantize_and_sat(acc_next) on last cycle; this is the same as final acc
    return quantize_and_sat(acc)


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


async def apply_and_check_one(
    dut, x, w, bias,
    NUM_INPUTS=8,
    X_W=8, W_W=8, B_W=32, OUT_W=16,
    X_FRAC=4, W_FRAC=4, B_FRAC=8, OUT_FRAC=8,
    GUARD_BITS=2,
    USE_RELU=1,
    max_wait_cycles=2000
):
    """Drive one transaction and check result."""
    dut.x_flat.value = pack_list_signed(x, X_W)
    dut.w_flat.value = pack_list_signed(w, W_W)
    dut.bias.value   = twos(bias, B_W)

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
    exp = model_serial(
        x, w, bias,
        NUM_INPUTS=NUM_INPUTS,
        X_W=X_W, W_W=W_W, B_W=B_W, OUT_W=OUT_W,
        X_FRAC=X_FRAC, W_FRAC=W_FRAC, B_FRAC=B_FRAC, OUT_FRAC=OUT_FRAC,
        GUARD_BITS=GUARD_BITS,
        USE_RELU=USE_RELU
    )

    if got != exp:
        # Helpful debug with float interpretations
        x_f = [fx_to_float(as_signed(v, X_W), X_FRAC) for v in x]
        w_f = [fx_to_float(as_signed(v, W_W), W_FRAC) for v in w]
        bias_f = fx_to_float(as_signed(bias, B_W), B_FRAC)
        got_f = fx_to_float(as_signed(got, OUT_W), OUT_FRAC)
        exp_f = fx_to_float(as_signed(exp, OUT_W), OUT_FRAC)
        raise AssertionError(
            "Mismatch:\n"
            f"  got={got} ({got_f})\n"
            f"  exp={exp} ({exp_f})\n"
            f"  x_fixed={x}\n"
            f"  w_fixed={w}\n"
            f"  bias_fixed={bias}\n"
            f"  x_real={x_f}\n"
            f"  w_real={w_f}\n"
            f"  bias_real={bias_f}\n"
        )


# ----------------------------
# Tests (expanded)
# ----------------------------
@cocotb.test()
async def test_known_vector_fractional(dut):
    """Directed test: fractional fixed-point values, no saturation, positive final sum."""
    cocotb.start_soon(generate_clock(dut))
    await reset_dut(dut)

    NUM_INPUTS = 8
    X_W = 8
    W_W = 8
    B_W = 32
    OUT_W = 16

    X_FRAC = 4
    W_FRAC = 4
    B_FRAC = 8
    OUT_FRAC = 8

    GUARD_BITS = 2
    USE_RELU = 1

    # Choose values that are exact multiples of 1/16 (for x,w) and 1/256 (for bias)
    x_r = [0.5, -1.25, 2.0, -0.75, 1.5, 0.25, -2.5, 0.0]
    w_r = [1.0, 0.5, -1.5, 2.0, -0.25, 1.75, 0.5, -1.0]
    bias_r = 0.5

    x = [fx_from_float(v, X_FRAC) for v in x_r]
    w = [fx_from_float(v, W_FRAC) for v in w_r]
    bias = fx_from_float(bias_r, B_FRAC)

    await apply_and_check_one(
        dut, x, w, bias,
        NUM_INPUTS,
        X_W, W_W, B_W, OUT_W,
        X_FRAC, W_FRAC, B_FRAC, OUT_FRAC,
        GUARD_BITS,
        USE_RELU
    )


@cocotb.test()
async def test_bias_only_fractional(dut):
    """Directed test: x=0 => output should equal bias (after bias align + quantize), with fractional bits."""
    cocotb.start_soon(generate_clock(dut))
    await reset_dut(dut)

    NUM_INPUTS = 8
    X_W = 8
    W_W = 8
    B_W = 32
    OUT_W = 16

    X_FRAC = 4
    W_FRAC = 4
    B_FRAC = 8
    OUT_FRAC = 8

    GUARD_BITS = 2
    USE_RELU = 1

    x = [0] * NUM_INPUTS
    w = [0] * NUM_INPUTS

    # Positive bias to avoid ReLU clamp
    bias_r = 1.25  # exact (320/256)
    bias = fx_from_float(bias_r, B_FRAC)

    await apply_and_check_one(
        dut, x, w, bias,
        NUM_INPUTS,
        X_W, W_W, B_W, OUT_W,
        X_FRAC, W_FRAC, B_FRAC, OUT_FRAC,
        GUARD_BITS,
        USE_RELU
    )


@cocotb.test()
async def test_relu_clamp_to_zero_fractional(dut):
    """Directed test: negative fixed-point accumulation should clamp to 0 when USE_RELU=1."""
    cocotb.start_soon(generate_clock(dut))
    await reset_dut(dut)

    NUM_INPUTS = 8
    X_W = 8
    W_W = 8
    B_W = 32
    OUT_W = 16

    X_FRAC = 4
    W_FRAC = 4
    B_FRAC = 8
    OUT_FRAC = 8

    GUARD_BITS = 2
    USE_RELU = 1

    # Make sure result is negative
    x = [fx_from_float(-2.0, X_FRAC)] * NUM_INPUTS
    w = [fx_from_float(3.0, W_FRAC)] * NUM_INPUTS
    bias = fx_from_float(0.0, B_FRAC)

    await apply_and_check_one(
        dut, x, w, bias,
        NUM_INPUTS,
        X_W, W_W, B_W, OUT_W,
        X_FRAC, W_FRAC, B_FRAC, OUT_FRAC,
        GUARD_BITS,
        USE_RELU
    )


@cocotb.test()
async def test_positive_saturation(dut):
    """Directed test: force positive overflow and confirm saturation to +32767 for OUT_W=16 (Q8 => ~+127.996)."""
    cocotb.start_soon(generate_clock(dut))
    await reset_dut(dut)

    NUM_INPUTS = 8
    X_W = 8
    W_W = 8
    B_W = 32
    OUT_W = 16

    X_FRAC = 4
    W_FRAC = 4
    B_FRAC = 8
    OUT_FRAC = 8

    GUARD_BITS = 2
    USE_RELU = 1

    # Maximum positive representable for Q4 in 8-bit signed is 127/16 = 7.9375
    x = [127] * NUM_INPUTS
    w = [127] * NUM_INPUTS
    bias = 0

    await apply_and_check_one(
        dut, x, w, bias,
        NUM_INPUTS,
        X_W, W_W, B_W, OUT_W,
        X_FRAC, W_FRAC, B_FRAC, OUT_FRAC,
        GUARD_BITS,
        USE_RELU
    )


@cocotb.test()
async def test_back_to_back_transactions(dut):
    """Two operations back-to-back, relying on in_ready (~busy) gating."""
    cocotb.start_soon(generate_clock(dut))
    await reset_dut(dut)

    NUM_INPUTS = 8
    X_W = 8
    W_W = 8
    B_W = 32
    OUT_W = 16

    X_FRAC = 4
    W_FRAC = 4
    B_FRAC = 8
    OUT_FRAC = 8

    GUARD_BITS = 2
    USE_RELU = 1

    vecs = [
        (
            [fx_from_float(0.5, X_FRAC)] * NUM_INPUTS,
            [fx_from_float(0.5, W_FRAC)] * NUM_INPUTS,
            fx_from_float(0.0, B_FRAC),
        ),
        (
            [fx_from_float(1.0, X_FRAC), fx_from_float(-1.0, X_FRAC)] * (NUM_INPUTS // 2),
            [fx_from_float(1.5, W_FRAC)] * NUM_INPUTS,
            fx_from_float(0.25, B_FRAC),
        ),
    ]

    for x, w, bias in vecs:
        await apply_and_check_one(
            dut, x, w, bias,
            NUM_INPUTS,
            X_W, W_W, B_W, OUT_W,
            X_FRAC, W_FRAC, B_FRAC, OUT_FRAC,
            GUARD_BITS,
            USE_RELU
        )


@cocotb.test()
async def test_random_regression_small(dut):
    """
    Random regression (small magnitudes): tries to avoid constant saturation and
    focuses on fixed-point correctness.
    """
    cocotb.start_soon(generate_clock(dut))
    await reset_dut(dut)

    random.seed(2)

    NUM_INPUTS = 8
    X_W = 8
    W_W = 8
    B_W = 32
    OUT_W = 16

    X_FRAC = 4
    W_FRAC = 4
    B_FRAC = 8
    OUT_FRAC = 8

    GUARD_BITS = 2
    USE_RELU = 1

    # Keep values small in raw fixed units.
    # Q4: +/-32 => +/-2.0 ; Q8 bias: +/-512 => +/-2.0
    for _ in range(100):
        x = [random.randint(-32, 31) for _ in range(NUM_INPUTS)]
        w = [random.randint(-32, 31) for _ in range(NUM_INPUTS)]
        bias = random.randint(-512, 511)

        await apply_and_check_one(
            dut, x, w, bias,
            NUM_INPUTS,
            X_W, W_W, B_W, OUT_W,
            X_FRAC, W_FRAC, B_FRAC, OUT_FRAC,
            GUARD_BITS,
            USE_RELU
        )


@cocotb.test()
async def test_random_regression_full_range(dut):
    """
    Random regression (full width ranges): includes cases that trigger saturation
    and also exercises the RTL's internal wrap/truncation behavior.
    """
    cocotb.start_soon(generate_clock(dut))
    await reset_dut(dut)

    random.seed(7)

    NUM_INPUTS = 8
    X_W = 8
    W_W = 8
    B_W = 32
    OUT_W = 16

    X_FRAC = 4
    W_FRAC = 4
    B_FRAC = 8
    OUT_FRAC = 8

    GUARD_BITS = 2
    USE_RELU = 1

    for _ in range(50):
        x = [random.randint(-(1 << (X_W - 1)), (1 << (X_W - 1)) - 1) for _ in range(NUM_INPUTS)]
        w = [random.randint(-(1 << (W_W - 1)), (1 << (W_W - 1)) - 1) for _ in range(NUM_INPUTS)]
        # Bias is 32-bit in RTL; still randomize within its representable range.
        bias = random.randint(-(1 << (B_W - 1)), (1 << (B_W - 1)) - 1)

        await apply_and_check_one(
            dut, x, w, bias,
            NUM_INPUTS,
            X_W, W_W, B_W, OUT_W,
            X_FRAC, W_FRAC, B_FRAC, OUT_FRAC,
            GUARD_BITS,
            USE_RELU
        )


def test_neuron_mac_serial_hidden_runner():
    sim = os.getenv("SIM", "icarus")
    proj_path = Path(__file__).resolve().parent.parent

    sources = [
        proj_path / "sources" / "neuron_mac_serial.v",
    ]

    runner = get_runner(sim)
    runner.build(
        sources=sources,
        hdl_toplevel="neuron_mac_serial",
        always=True,
    )
    runner.test(
        hdl_toplevel="neuron_mac_serial",
        test_module="test_neuron_mac_serial_hidden",
    )
