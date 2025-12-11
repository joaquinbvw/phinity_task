# neuron_mac_simple — Specification

## Overview
`neuron_mac_simple` implements a simplified artificial neuron / MAC engine:

\[
y = bias + \sum_{i=0}^{NUM\_INPUTS-1} x[i]\cdot w[i]
\]

- Inputs `x[i]` and `w[i]` are signed integers (two’s complement).
- The module processes one input vector at a time (no pipelining).
- Optional ReLU can be enabled to clamp negative results to 0.
- Final result is saturated to signed `OUT_W` bits.

This design does **not** implement fractional alignment, scaling, or rounding. It is pure integer MAC + optional ReLU + saturation.

---

## Module Interface

### Ports
- `clk` (input): rising-edge clock.
- `rst_n` (input): active-low async reset.

#### Input handshake
- `in_valid` (input): asserts that inputs for a new transaction are present.
- `in_ready` (output): high when the module can accept a new transaction.

A new transaction is accepted on a rising clock edge when:
- `in_valid == 1` AND `in_ready == 1`

#### Inputs
- `bias` (input, signed, `B_W` bits): initial accumulator value.
- `x_flat` (input, `NUM_INPUTS*X_W` bits): packed vector of `x[i]`.
- `w_flat` (input, `NUM_INPUTS*W_W` bits): packed vector of `w[i]`.

#### Outputs
- `out_valid` (output): pulses high for **one clock cycle** when `out_data` is valid.
- `out_data` (output, signed, `OUT_W` bits): saturated final neuron output.
- `busy` (output): high while MAC processing is in progress.

---

## Parameters
- `NUM_INPUTS` (integer, default 8): number of `x[i], w[i]` pairs.
- `X_W` (integer, default 8): width of each signed input sample `x[i]`.
- `W_W` (integer, default 8): width of each signed weight `w[i]`.
- `B_W` (integer, default 16): width of signed bias.
- `OUT_W` (integer, default 16): width of signed output.
- `GUARD_BITS` (integer, default 2): extra accumulator bits to reduce overflow risk.
- `USE_RELU` (integer, default 1): if nonzero, apply ReLU before saturation.

---

## Data Packing (x_flat / w_flat)
`x_flat` and `w_flat` are packed little-endian by element index.

For `i = 0..NUM_INPUTS-1`:

- `x[i]` occupies bits: `x_flat[(i*X_W) +: X_W]`
- `w[i]` occupies bits: `w_flat[(i*W_W) +: W_W]`

In particular:
- `x[0]` is in `x_flat[X_W-1:0]`
- `w[0]` is in `w_flat[W_W-1:0]`

All values are interpreted as **two’s complement signed**.

---

## Functional Behavior

### Reset
When `rst_n == 0`:
- `busy = 0`
- `in_ready = 1`
- `out_valid = 0`
- `out_data = 0`
- internal state is cleared.

### Accepting a transaction
On a rising edge where `in_valid && in_ready`:
1. Internal accumulator `acc` is loaded with `bias` (sign-extended).
2. Internal copies of `x_flat` and `w_flat` are captured.
3. `busy` becomes 1.
4. The internal element index counter starts at 0.

### MAC processing
While `busy == 1`, the module performs **one multiply-add per clock cycle**:

For each `k = 0..NUM_INPUTS-1` in order:
- `prod_k = x[k] * w[k]` (signed multiplication)
- `acc := acc + prod_k`

After the final multiply-add (`k = NUM_INPUTS-1`), the module produces the output and ends processing.

### ReLU (optional)
If `USE_RELU != 0`, apply ReLU to the final accumulated value:
- If `acc < 0`, use `acc_relu = 0`
- Else `acc_relu = acc`

If `USE_RELU == 0`, then `acc_relu = acc`.

### Saturation to OUT_W
The final output is saturated to signed `OUT_W` range:

- `OUT_MAX = 2^(OUT_W-1) - 1`
- `OUT_MIN = -2^(OUT_W-1)`

Output value:
- If `acc_relu > OUT_MAX` → `out_data = OUT_MAX`
- Else if `acc_relu < OUT_MIN` → `out_data = OUT_MIN`
- Else `out_data = acc_relu` truncated to `OUT_W` bits.

---

## Handshake & Timing

### Input readiness
- `in_ready = 1` only when `busy = 0`.
- While `busy = 1`, `in_ready = 0` and new inputs are ignored.

### Latency
Let a transaction be accepted at clock edge **T0**.

- The module performs one MAC per cycle for `NUM_INPUTS` cycles.
- `out_valid` is asserted for **one cycle** on edge **T0 + NUM_INPUTS**.
- `out_data` is valid when `out_valid == 1`.

### Throughput / bubble
The module is not pipelined. It cannot accept a new transaction on the same edge it asserts `out_valid`.
The earliest next accept can occur is the next clock edge after `out_valid` (i.e., there is a one-cycle bubble between transactions).

---

## Worked Example (default widths)
Assume:
- `NUM_INPUTS=2`, `X_W=8`, `W_W=8`, `B_W=16`, `OUT_W=16`, `USE_RELU=1`
- `x = [10, -3]`
- `w = [7, 20]`
- `bias = 100`

Compute:
- `acc = 100`
- `acc += 10*7 = 70` → 170
- `acc += (-3)*20 = -60` → 110
- ReLU: 110 stays 110
- Saturation: 110 is within int16, so `out_data = 110`.

Packing:
- `x_flat = x[0] | (x[1] << 8)`
- `w_flat = w[0] | (w[1] << 8)`
