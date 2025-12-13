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


# ----------------------------
# Parameter introspection helpers
# ----------------------------

def _get_param_int(dut, name, default):
    if hasattr(dut, name):
        obj = getattr(dut, name)
        try:
            return int(obj)
        except Exception:
            try:
                return int(obj.value)
            except Exception:
                return default
    return default


def get_dut_params(dut):
    """
    Read generics/parameters from the DUT if available, otherwise fall back
    to the default values used in the spec.
    """
    params = {}
    params["NUM_INPUTS"] = _get_param_int(dut, "NUM_INPUTS", 8)
    params["X_W"]        = _get_param_int(dut, "X_W", 8)
    params["W_W"]        = _get_param_int(dut, "W_W", 8)
    params["B_W"]        = _get_param_int(dut, "B_W", 32)
    params["OUT_W"]      = _get_param_int(dut, "OUT_W", 16)
    params["X_FRAC"]     = _get_param_int(dut, "X_FRAC", 4)
    params["W_FRAC"]     = _get_param_int(dut, "W_FRAC", 4)
    params["B_FRAC"]     = _get_param_int(dut, "B_FRAC", 8)
    params["OUT_FRAC"]   = _get_param_int(dut, "OUT_FRAC", 8)
    params["GUARD_BITS"] = _get_param_int(dut, "GUARD_BITS", 2)
    params["USE_RELU"]   = _get_param_int(dut, "USE_RELU", 1)
    return params


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
        # We expect in_ready to be ~busy at all times
        if int(dut.busy.value) == 1:
            assert int(dut.in_ready.value) == 0, "in_ready must be 0 while busy"
        else:
            assert int(dut.in_ready.value) == 1, "in_ready must be 1 when not busy"

        if int(dut.in_ready.value) == 1:
            break
        await RisingEdge(dut.clk)
    else:
        raise AssertionError("Timeout waiting for in_ready==1")

    # Pulse in_valid for one cycle
    dut.in_valid.value = 1
    await RisingEdge(dut.clk)
    dut.in_valid.value = 0

    # Wait for transaction to complete, checking handshake behaviour
    saw_out_valid = False
    got = None

    for _ in range(max_wait_cycles):
        # in_ready must always be the complement of busy
        if int(dut.busy.value) == 1:
            assert int(dut.in_ready.value) == 0, "in_ready must be 0 while busy"
        else:
            assert int(dut.in_ready.value) == 1, "in_ready must be 1 when not busy"

        if int(dut.out_valid.value) == 1:
            # Capture output on the cycle out_valid is asserted
            saw_out_valid = True
            got = read_signed(dut.out_data)

            # One-cycle pulse & return to idle on the *next* clock
            await RisingEdge(dut.clk)
            assert int(dut.out_valid.value) == 0, "out_valid must be a one-cycle pulse"
            assert int(dut.busy.value) == 0, "busy must deassert after result"
            assert int(dut.in_ready.value) == 1, "in_ready must re-assert after result"
            break

        await RisingEdge(dut.clk)

    if not saw_out_valid:
        raise AssertionError("Timeout waiting for out_valid==1")

    # Now compare against the model
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

    params = get_dut_params(dut)
    NUM_INPUTS = params["NUM_INPUTS"]
    X_W       = params["X_W"]
    W_W       = params["W_W"]
    B_W       = params["B_W"]
    OUT_W     = params["OUT_W"]
    X_FRAC    = params["X_FRAC"]
    W_FRAC    = params["W_FRAC"]
    B_FRAC    = params["B_FRAC"]
    OUT_FRAC  = params["OUT_FRAC"]
    GUARD_BITS = params["GUARD_BITS"]
    USE_RELU   = params["USE_RELU"]

    # Choose values that are exact multiples of 1/2^X_FRAC (for x,w) and 1/2^B_FRAC (for bias)
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

    params = get_dut_params(dut)
    NUM_INPUTS = params["NUM_INPUTS"]
    X_W       = params["X_W"]
    W_W       = params["W_W"]
    B_W       = params["B_W"]
    OUT_W     = params["OUT_W"]
    X_FRAC    = params["X_FRAC"]
    W_FRAC    = params["W_FRAC"]
    B_FRAC    = params["B_FRAC"]
    OUT_FRAC  = params["OUT_FRAC"]
    GUARD_BITS = params["GUARD_BITS"]
    USE_RELU   = params["USE_RELU"]

    x = [0] * NUM_INPUTS
    w = [0] * NUM_INPUTS

    # Positive bias to avoid ReLU clamp
    bias_r = 1.25
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

    params = get_dut_params(dut)
    NUM_INPUTS = params["NUM_INPUTS"]
    X_W       = params["X_W"]
    W_W       = params["W_W"]
    B_W       = params["B_W"]
    OUT_W     = params["OUT_W"]
    X_FRAC    = params["X_FRAC"]
    W_FRAC    = params["W_FRAC"]
    B_FRAC    = params["B_FRAC"]
    OUT_FRAC  = params["OUT_FRAC"]
    GUARD_BITS = params["GUARD_BITS"]
    USE_RELU   = params["USE_RELU"]

    # Make sure result is negative before ReLU
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
    """Directed test: force positive overflow and confirm saturation to OUT_MAX."""
    cocotb.start_soon(generate_clock(dut))
    await reset_dut(dut)

    params = get_dut_params(dut)
    NUM_INPUTS = params["NUM_INPUTS"]
    X_W       = params["X_W"]
    W_W       = params["W_W"]
    B_W       = params["B_W"]
    OUT_W     = params["OUT_W"]
    X_FRAC    = params["X_FRAC"]
    W_FRAC    = params["W_FRAC"]
    B_FRAC    = params["B_FRAC"]
    OUT_FRAC  = params["OUT_FRAC"]
    GUARD_BITS = params["GUARD_BITS"]
    USE_RELU   = params["USE_RELU"]

    # Drive near-maximum positive values
    max_x = (1 << (X_W - 1)) - 1
    max_w = (1 << (W_W - 1)) - 1
    x = [max_x] * NUM_INPUTS
    w = [max_w] * NUM_INPUTS
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

    params = get_dut_params(dut)
    NUM_INPUTS = params["NUM_INPUTS"]
    X_W       = params["X_W"]
    W_W       = params["W_W"]
    B_W       = params["B_W"]
    OUT_W     = params["OUT_W"]
    X_FRAC    = params["X_FRAC"]
    W_FRAC    = params["W_FRAC"]
    B_FRAC    = params["B_FRAC"]
    OUT_FRAC  = params["OUT_FRAC"]
    GUARD_BITS = params["GUARD_BITS"]
    USE_RELU   = params["USE_RELU"]

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


#@cocotb.test()
#async def test_in_valid_held_high_two_ops(dut):
#    """
#    Hold in_valid high continuously and rely on in_ready/~busy gating to
#    accept exactly two back-to-back operations with different vectors.
#    """
#    cocotb.start_soon(generate_clock(dut))
#    await reset_dut(dut)
#
#    params = get_dut_params(dut)
#    NUM_INPUTS = params["NUM_INPUTS"]
#    X_W       = params["X_W"]
#    W_W       = params["W_W"]
#    B_W       = params["B_W"]
#    OUT_W     = params["OUT_W"]
#    X_FRAC    = params["X_FRAC"]
#    W_FRAC    = params["W_FRAC"]
#    B_FRAC    = params["B_FRAC"]
#    OUT_FRAC  = params["OUT_FRAC"]
#    GUARD_BITS = params["GUARD_BITS"]
#    USE_RELU   = params["USE_RELU"]
#
#    # First operation: modest positive sum
#    x1 = [fx_from_float(0.5, X_FRAC)] * NUM_INPUTS
#    w1 = [fx_from_float(0.25, W_FRAC)] * NUM_INPUTS
#    bias1 = fx_from_float(0.0, B_FRAC)
#
#    # Second operation: different pattern, mixed signs
#    x2 = [fx_from_float(1.0, X_FRAC), fx_from_float(-0.5, X_FRAC)] * (NUM_INPUTS // 2)
#    w2 = [fx_from_float(0.75, W_FRAC)] * NUM_INPUTS
#    bias2 = fx_from_float(0.125, B_FRAC)
#
#    exp1 = model_serial(
#        x1, w1, bias1,
#        NUM_INPUTS=NUM_INPUTS,
#        X_W=X_W, W_W=W_W, B_W=B_W, OUT_W=OUT_W,
#        X_FRAC=X_FRAC, W_FRAC=W_FRAC, B_FRAC=B_FRAC, OUT_FRAC=OUT_FRAC,
#        GUARD_BITS=GUARD_BITS,
#        USE_RELU=USE_RELU,
#    )
#    exp2 = model_serial(
#        x2, w2, bias2,
#        NUM_INPUTS=NUM_INPUTS,
#        X_W=X_W, W_W=W_W, B_W=B_W, OUT_W=OUT_W,
#        X_FRAC=X_FRAC, W_FRAC=W_FRAC, B_FRAC=B_FRAC, OUT_FRAC=OUT_FRAC,
#        GUARD_BITS=GUARD_BITS,
#        USE_RELU=USE_RELU,
#    )
#
#    # Wait for in_ready before starting
#    while int(dut.in_ready.value) == 0:
#        await RisingEdge(dut.clk)
#
#    # Start first op, assert in_valid and keep it high
#    dut.x_flat.value = pack_list_signed(x1, X_W)
#    dut.w_flat.value = pack_list_signed(w1, W_W)
#    dut.bias.value   = twos(bias1, B_W)
#    dut.in_valid.value = 1
#
#    outputs = []
#    first_done = False
#    max_cycles = 5000
#
#    for _ in range(max_cycles):
#        # Handshake must always obey in_ready = ~busy
#        if int(dut.busy.value) == 1:
#            assert int(dut.in_ready.value) == 0, "in_ready must be 0 while busy"
#        else:
#            assert int(dut.in_ready.value) == 1, "in_ready must be 1 when not busy"
#
#        if int(dut.out_valid.value) == 1:
#            outputs.append(read_signed(dut.out_data))
#
#            if not first_done:
#                # First result just produced; prepare second op's inputs so that
#                # they are captured on the next cycle where in_ready goes high.
#                first_done = True
#                dut.x_flat.value = pack_list_signed(x2, X_W)
#                dut.w_flat.value = pack_list_signed(w2, W_W)
#                dut.bias.value   = twos(bias2, B_W)
#            else:
#                # Second result observed; one more cycle for post-conditions
#                await RisingEdge(dut.clk)
#                break
#
#        await RisingEdge(dut.clk)
#
#    # Drop in_valid
#    dut.in_valid.value = 0
#
#    assert len(outputs) == 2, "Expected two outputs with in_valid held high"
#    assert outputs[0] == exp1, f"First result mismatch: got {outputs[0]}, exp {exp1}"
#    assert outputs[1] == exp2, f"Second result mismatch: got {outputs[1]}, exp {exp2}"


@cocotb.test()
async def test_random_regression_small(dut):
    """
    Random regression (small magnitudes): tries to avoid constant saturation and
    focuses on fixed-point correctness.
    """
    cocotb.start_soon(generate_clock(dut))
    await reset_dut(dut)

    random.seed(2)

    params = get_dut_params(dut)
    NUM_INPUTS = params["NUM_INPUTS"]
    X_W       = params["X_W"]
    W_W       = params["W_W"]
    B_W       = params["B_W"]
    OUT_W     = params["OUT_W"]
    X_FRAC    = params["X_FRAC"]
    W_FRAC    = params["W_FRAC"]
    B_FRAC    = params["B_FRAC"]
    OUT_FRAC  = params["OUT_FRAC"]
    GUARD_BITS = params["GUARD_BITS"]
    USE_RELU   = params["USE_RELU"]

    # Keep values small in raw fixed units.
    # QX_FRAC: +/-32 => +/-2.0 ; QB_FRAC bias: +/-512 => +/-2.0 (for FRAC=4/8 default)
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
async def test_negative_saturation_no_relu(dut):
    """Force a large negative result and check saturation to OUT_MIN when USE_RELU=0.

    This is only meaningful when the DUT was instantiated/configured with USE_RELU=0.
    If we detect USE_RELU!=0 (or cannot read it), we log and return.
    """
    # Try to read USE_RELU parameter from the DUT if available
    use_relu_param = None
    if hasattr(dut, "USE_RELU"):
        try:
            try:
                use_relu_param = int(dut.USE_RELU)
            except Exception:
                use_relu_param = int(dut.USE_RELU.value)
        except Exception:
            use_relu_param = None

    if use_relu_param is None:
        dut._log.info(
            "Skipping test_negative_saturation_no_relu: unable to read USE_RELU parameter; "
            "assuming USE_RELU=1 in this build."
        )
        return

    if use_relu_param != 0:
        dut._log.info(
            "Skipping test_negative_saturation_no_relu: USE_RELU=%d (need USE_RELU=0).",
            use_relu_param,
        )
        return

    # If we got here, DUT is configured with USE_RELU=0
    cocotb.start_soon(generate_clock(dut))
    await reset_dut(dut)

    params = get_dut_params(dut)
    NUM_INPUTS = params["NUM_INPUTS"]
    X_W       = params["X_W"]
    W_W       = params["W_W"]
    B_W       = params["B_W"]
    OUT_W     = params["OUT_W"]
    X_FRAC    = params["X_FRAC"]
    W_FRAC    = params["W_FRAC"]
    B_FRAC    = params["B_FRAC"]
    OUT_FRAC  = params["OUT_FRAC"]
    GUARD_BITS = params["GUARD_BITS"]
    USE_RELU   = 0   # Model config

    # Strong negative sum
    x = [fx_from_float(-4.0, X_FRAC)] * NUM_INPUTS
    w = [fx_from_float(4.0, W_FRAC)] * NUM_INPUTS
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
async def test_async_reset_while_busy(dut):
    """Assert rst_n while MAC is running; DUT must cleanly return to idle."""
    cocotb.start_soon(generate_clock(dut))
    await reset_dut(dut)

    params = get_dut_params(dut)
    NUM_INPUTS = params["NUM_INPUTS"]
    X_W       = params["X_W"]
    W_W       = params["W_W"]
    B_W       = params["B_W"]
    OUT_W     = params["OUT_W"]
    X_FRAC    = params["X_FRAC"]
    W_FRAC    = params["W_FRAC"]
    B_FRAC    = params["B_FRAC"]
    OUT_FRAC  = params["OUT_FRAC"]
    GUARD_BITS = params["GUARD_BITS"]
    USE_RELU   = params["USE_RELU"]

    # Simple non-zero vector
    x = [fx_from_float(0.5, X_FRAC)] * NUM_INPUTS
    w = [fx_from_float(0.5, W_FRAC)] * NUM_INPUTS
    bias = fx_from_float(0.0, B_FRAC)

    # Drive first transaction manually (we're going to kill it mid-flight)
    dut.x_flat.value = pack_list_signed(x, X_W)
    dut.w_flat.value = pack_list_signed(w, W_W)
    dut.bias.value   = twos(bias, B_W)

    # Wait for in_ready
    while int(dut.in_ready.value) == 0:
        await RisingEdge(dut.clk)

    dut.in_valid.value = 1
    await RisingEdge(dut.clk)
    dut.in_valid.value = 0

    # Let it run a few cycles so busy should be asserted
    for _ in range(3):
        await RisingEdge(dut.clk)

    assert int(dut.busy.value) == 1, "DUT should be busy before mid-op reset"

    # Assert async reset
    dut.rst_n.value = 0
    await RisingEdge(dut.clk)

    # After reset, must be fully idle
    assert int(dut.busy.value) == 0, "busy must drop after reset"
    assert int(dut.out_valid.value) == 0, "out_valid must be 0 after reset"
    assert int(dut.in_ready.value) == 1, "in_ready must be 1 after reset"

    # Release reset and make sure a new op still works
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

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

    params = get_dut_params(dut)
    NUM_INPUTS = params["NUM_INPUTS"]
    X_W       = params["X_W"]
    W_W       = params["W_W"]
    B_W       = params["B_W"]
    OUT_W     = params["OUT_W"]
    X_FRAC    = params["X_FRAC"]
    W_FRAC    = params["W_FRAC"]
    B_FRAC    = params["B_FRAC"]
    OUT_FRAC  = params["OUT_FRAC"]
    GUARD_BITS = params["GUARD_BITS"]
    USE_RELU   = params["USE_RELU"]

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
