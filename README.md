# neuron_mac_serial — RTL Code Completion Task (Fixed-Point Neuron MAC)

## 1. Why I chose this task

I chose this task because it is a realistic “small-but-nontrivial” RTL unit that forces an agent to reason across:
- fixed-point scaling and deterministic rounding,
- multi-cycle sequencing (serial MAC across `NUM_INPUTS`),
- correctness under masking (sparsity) and selectable activations,
- a simple valid/ready handshake with strict timing expectations.

It’s compact enough to fit as a single-module code-completion problem, but still difficult in the ways RTL often is: cycle-accurate behavior, signed arithmetic corner cases, and fixed-point quantization/saturation rules.

## 2. Industry relevance

This design mirrors common building blocks used in practical hardware/embedded ML inference and DSP pipelines:
- **Fixed-point compute** is a default choice for resource/latency/power efficiency in FPGA/ASIC accelerators.
- **Serial MAC** maps well to resource-constrained implementations (area-first designs) and is a frequent baseline micro-architecture.
- **Sparsity masking** reflects real workloads (pruning, structured sparsity, conditional compute / gating).
- **Activation selection** (identity / ReLU / leaky ReLU / clamp) is representative of typical inference datapaths and bounded-activation stabilization patterns.
- **Saturation + deterministic rounding** are exactly the kinds of implementation details that cause real silicon/FPGA mismatches if not handled precisely.

## 3. Brief context of the codebase

This repository is organized as a self-contained RTL task with:
- a conceptual specification describing the neuron math and fixed-point rules,
- a Verilog module to complete,
- a cocotb-based test suite with a bit-accurate Python reference model.

### 3.1 Repository layout (recommended)

- `docs/Specification.md`  
  Conceptual spec: fixed-point encoding, bias alignment, masking, activation behavior, quantization (with deterministic right-shift rounding), and output saturation.

- `sources/neuron_mac_serial.v`  
  **Task RTL** (code completion): Verilog-2001 module `neuron_mac_serial` with the missing implementation region.

- `tests/test_neuron_mac_serial_hidden.py`  
  cocotb tests + a bit-accurate Python model (`model_serial`) used to validate functionality across:
  - directed fixed-point cases,
  - saturation cases,
  - reset/handshake behavior,
  - randomized regression,
  - activation modes + sparsity masking.

- `prompt.txt` (or `docs/Prompt.txt`)  
  The exact prompt given to the agent (more implementation-specific about ports, packing, and handshake).

> Notes:
> - The testbench assumes little-endian packing for `x_flat` / `w_flat`: element 0 is in LSBs, element `i` is slice `i*W +: W`.
> - Handshake requirement: `in_ready` must be high exactly when idle, i.e. `in_ready = ~busy`, and `out_valid` must be a one-cycle pulse.

### 3.2 How to run locally (cocotb Makefile flow)

I ran the cocotb tests locally using a dedicated Makefile that targets Icarus Verilog and supports waveform dumping via `WAVES=1`.

Example Makefile:

```make
# Makefile.serial
SIM ?= icarus
TOPLEVEL_LANG ?= verilog
VERILOG_SOURCES = $(PWD)/neuron_mac_serial.v
TOPLEVEL = neuron_mac_serial
MODULE = test_neuron_mac_serial
include $(shell cocotb-config --makefiles)/Makefile.sim
````

Run command (with waveform dumping enabled):

```bash
make -f Makefile.serial SIM=icarus WAVES=1
```

## 4. Latest evaluation results

> IMPORTANT: HUD link must remain private (do not publish).

* **HUD run link (private):** `https://www.hud.ai/jobs/7b0b9e12-a5ee-47a1-9bfd-32075ebf2f32`
* **Model / config:** `claude-sonnet-4-5-20250929 --max-steps 80`
* **Score / Pass@10:** `40%`
* **Notes:**

  * This version landed in a ~40% pass@10 range.
  * While tuning difficulty, I also implemented a *harder* variant that introduced an **external sequential multiplier submodule** (to increase multi-cycle control complexity). That variant evaluated to **0% pass@10**, and progressing it into the target range would have required additional iteration time. For the submission, I reverted to this current version as a better-balanced task.

## 5. What the agent is expected to do (one-paragraph summary)

Given the conceptual spec in `docs/Specification.md`, the agent must implement a synthesizable Verilog-2001 serial neuron MAC that:

* latches packed inputs/weights/bias/mask and activation on `in_valid && in_ready`,
* accumulates one masked product per cycle over `NUM_INPUTS`,
* applies the selected activation in the accumulator domain,
* converts from internal fractional scale to `OUT_FRAC` using the deterministic rounding rule,
* saturates to the signed `OUT_W` range,
* asserts `out_valid` for exactly one cycle with `out_data`, then returns idle.
