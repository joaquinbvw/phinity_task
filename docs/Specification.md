# Artificial Neuron MAC (Fixed-Point, Serial) — Specification

## 1. Overview

Artificial neurons are basic computational units of modern neural networks. Conceptually, each neuron receives a set of input values, multiplies each input by an associated weight, adds all these contributions together, and then adds a bias term. The result of this weighted sum is often passed through an activation function, which introduces nonlinearity and helps the network approximate complex relationships.

In a typical setting, the inputs represent features of a data instance. Each weight encodes how strongly a particular input contributes to the neuron's decision: a positive weight reinforces the input, a negative weight suppresses it, and a weight near zero means the input has little influence. The bias term acts as a baseline offset, shifting the neuron’s decision threshold independently of the inputs.

Three concepts are central to this behavior:

- **Bias**: a constant term added to the weighted sum of inputs. It allows the neuron to activate even when all inputs are zero and shifts the decision threshold.
- **Activation function**: a (usually nonlinear) function applied to the weighted sum plus bias. A common choice is ReLU.
- **Saturation**: a mechanism that limits the output to lie within a representable range, preventing unbounded growth.

A widely used activation in modern deep learning is the **Rectified Linear Unit (ReLU)**. ReLU sets all negative inputs to zero and passes positive inputs unchanged.

While the conceptual description of neurons is typically expressed in real numbers and floating-point arithmetic, many embedded and hardware-oriented implementations use **fixed-point arithmetic** to achieve efficient, deterministic, and low-power operation. In fixed-point representations, real-valued quantities are mapped to integers by assuming a fixed binary point position. For example, a real number $r$ might be represented as an integer $q$ such that:

$$
r \approx \frac{q}{2^F}
$$

where $F$ is the number of fractional bits.

Fixed-point arithmetic introduces quantization effects and requires explicit handling of scaling, rounding, and saturation. This specification defines a neuron model at the conceptual level that:
- represents inputs/weights/bias/output as signed fixed-point values (two’s complement integers with an implicit binary point),
- defines how binary-point alignment is performed (especially for the bias),
- defines a deterministic rounding rule when reducing precision, and
- defines explicit output saturation.

---

## 2. Mathematical Behavior

### 2.1 Real-Valued Neuron (Conceptual)

Let:
- $N$ be the number of inputs,
- $x_0, x_1, \dots, x_{N-1}$ be real-valued inputs,
- $w_0, w_1, \dots, w_{N-1}$ be real-valued weights,
- $b$ be a real-valued bias.

The pre-activation is:

$$
s = b + \sum_{i=0}^{N-1} x_i \cdot w_i
$$

For ReLU activation:

$$
y_{\text{real}} = \max(0, s)
$$

If ReLU is disabled, then $y_{\text{real}} = s$.

### 2.2 Fixed-Point Encoding

Each quantity is represented as a signed two’s-complement integer with an implicit binary point:

- Input $x_i$ is represented by an integer $X_i$ with $F_x$ fractional bits:

$$
x_i \approx \frac{X_i}{2^{F_x}}
$$

- Weight $w_i$ is represented by an integer $W_i$ with $F_w$ fractional bits:

$$
w_i \approx \frac{W_i}{2^{F_w}}
$$

- Bias $b$ is represented by an integer $B$ with $F_b$ fractional bits:

$$
b \approx \frac{B}{2^{F_b}}
$$

- Output $y$ is represented by an integer $Y$ with $F_y$ fractional bits:

$$
y \approx \frac{Y}{2^{F_y}}
$$

A product $X_i \cdot W_i$ naturally has:

$$
F_p = F_x + F_w
$$

fractional bits. It is convenient to interpret the accumulator in this same “product scale” (i.e., with $F_p$ fractional bits).

### 2.3 Fixed-Point Computation (Integer Domain)

The computation is defined in integer form while tracking the implied binary-point positions.

#### 2.3.1 Bias Alignment (to product scale)

The bias integer $B$ (with $F_b$ fractional bits) is aligned into the accumulator scale (with $F_p$ fractional bits), producing $B_{\text{aligned}}$.

- If $F_b > F_p$, shift right by $sh = F_b - F_p$ with rounding (rule below).
- If $F_p > F_b$, shift left by $sh = F_p - F_b$.
- If $F_b = F_p$, no shift is required.

**Deterministic rounding rule for right shifts** (used whenever shifting right by $sh \ge 1$):

1. Compute $c = 2^{sh-1}$.
2. If the value is nonnegative, add $c$ before shifting.
3. If the value is negative, subtract $c$ before shifting.
4. Then perform an arithmetic right shift by $sh$.

This rule is intentionally stated as an algorithm (rather than labeled as “round-to-nearest”), since its behavior for negative values depends on two’s-complement arithmetic shifting.

#### 2.3.2 Multiply–Accumulate

Initialize the integer accumulator:

$$
ACC_0 = B_{\text{aligned}}
$$

Then accumulate products:

$$
ACC_{k+1} = ACC_k + (X_k \cdot W_k), \quad k = 0,1,\dots,N-1
$$

After all inputs:

$$
ACC_{\text{final}} = B_{\text{aligned}} + \sum_{i=0}^{N-1} (X_i \cdot W_i)
$$

#### 2.3.3 Optional ReLU (in accumulator scale)

If ReLU is enabled:

$$
ACC_{\text{relu}} =
\begin{cases}
0, & \text{if } ACC_{\text{final}} < 0 \\
ACC_{\text{final}}, & \text{otherwise}
\end{cases}
$$

If ReLU is disabled:

$$
ACC_{\text{relu}} = ACC_{\text{final}}
$$

#### 2.3.4 Output Quantization (product scale → output scale)

Convert from $F_p$ fractional bits to $F_y$ fractional bits:

- If $F_p > F_y$, shift right by $sh = F_p - F_y$ using the same deterministic rounding rule as above.
- If $F_y > F_p$, shift left by $sh = F_y - F_p$.
- If $F_p = F_y$, no shift.

Let the quantized integer be $Q$.

#### 2.3.5 Output Saturation (finite signed range)

The output integer $Y$ is obtained by saturating $Q$ to the signed range representable by the chosen output width $W_y$:

$$
Y_{\min} = -2^{W_y-1}, \quad Y_{\max} = 2^{W_y-1}-1
$$

$$
Y =
\begin{cases}
Y_{\max}, & \text{if } Q > Y_{\max} \\
Y_{\min}, & \text{if } Q < Y_{\min} \\
Q, & \text{otherwise}
\end{cases}
$$

---

## 3. Bit-Width and Overflow Notes (Implementation-Relevant)

This section captures common fixed-point implementation consequences when using finite-width signed arithmetic.

### 3.1 Typical width growth

If inputs use $W_x$ bits and weights use $W_w$ bits (both signed), the full-precision product typically needs:

$$
W_p = W_x + W_w
$$

bits (signed). When summing $N$ products, a common rule-of-thumb for additional headroom is approximately $\lceil \log_2(N) \rceil$ bits, plus any explicit guard bits to reduce wraparound risk:

$$
W_{\text{acc}} \approx W_p + \left\lceil \log_2(N) \right\rceil + G
$$

where $G$ is a chosen number of guard bits.

### 3.2 Wraparound vs saturation

In many hardware-friendly fixed-point designs:
- internal arithmetic is performed in a fixed width and may **wrap around** on overflow (two’s-complement modular behavior), and
- only the final output is **explicitly saturated** to the output width’s signed range.

Practical implication: scaling choices (fractional bits) and accumulator width must be chosen so that internal wraparound is either extremely unlikely under expected workloads, or explicitly acceptable.

---

## 4. Working Examples

### Example A — Fractional fixed-point, no scale changes

Assume:
- $N = 2$
- $F_x = 4$, $F_w = 4$ so $F_p = 8$
- $F_b = 8$ (bias already aligned to product scale)
- $F_y = 8$ (no output scale change)
- ReLU enabled and no saturation occurs.

Real values:
- $x = [0.5, -1.25]$
- $w = [1.0, 0.5]$
- $b = 0.5$

Integer encodings:
- $X = [0.5 \cdot 2^4, -1.25 \cdot 2^4] = [8, -20]$
- $W = [1.0 \cdot 2^4, 0.5 \cdot 2^4] = [16, 8]$
- $B = 0.5 \cdot 2^8 = 128$

Integer-domain computation (accumulator interpreted at $F_p = 8$ fractional bits):
- Products: $8 \cdot 16 = 128$, and $-20 \cdot 8 = -160$
- Accumulate: $ACC = 128 + 128 - 160 = 96$
- ReLU: unchanged (nonnegative)
- Output scale: unchanged, so $Y = 96$

Interpretation:

$$
y \approx \frac{96}{2^8} = 0.375
$$

which matches:

$$
0.5 + (0.5 \cdot 1.0) + (-1.25 \cdot 0.5) = 0.375
$$

### Example B — Demonstrating deterministic rounding on a right shift

Assume a right shift by $sh = 2$ is needed when reducing fractional precision.

Let the integer value be $v = 5$ (in some fixed-point scale). The deterministic rounding rule does:
- $c = 2^{sh-1} = 2$
- $v \ge 0$, so $v' = v + c = 7$
- $v_{\text{out}} = v' \gg 2 = 1$

This example illustrates the “add/subtract $2^{sh-1}$ then arithmetic shift” rounding algorithm used whenever precision is reduced via right shifts.

---

## 5. Summary of Required Behavior

A compliant implementation must:

1. Represent inputs, weights, bias, and output as signed fixed-point values (two’s-complement integers with an implicit binary point).
2. Interpret products in a scale with fractional bits $F_p = F_x + F_w$ and accumulate in that same scale.
3. Align the bias into the product/accumulator scale before accumulation (left shift if increasing fractional bits, right shift if decreasing fractional bits).
4. When right-shifting for scale reduction, apply the deterministic rounding rule: add $2^{sh-1}$ for nonnegative values or subtract $2^{sh-1}$ for negative values, then arithmetic shift right by $sh$.
5. Optionally apply ReLU by clamping negative accumulated results to zero before output quantization and saturation.
6. Quantize from the accumulator’s fractional scale to the output’s fractional scale using the same shift-and-round rule, then saturate the result to the signed output range.

