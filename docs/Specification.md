
# Artificial Neuron MAC — Specification

## 1. Introduction

Artificial neurons are the basic computational units of modern neural networks. Conceptually, each neuron receives a set of input values, multiplies each input by an associated weight, adds all these contributions together, and then adds a bias term. The result of this weighted sum is then passed through an activation function, which introduces nonlinearity and helps the network approximate complex relationships.

In a typical setting, the inputs can represent features of some data point. For example, in an image classification task, inputs to a neuron in a later layer might represent higher-level visual features detected in the image. Each weight encodes how strongly a particular input contributes to the neuron's decision: a positive weight reinforces the input, a negative weight suppresses it, and a weight near zero means the input has little influence. The bias term can be seen as a baseline offset for the neuron's activation, allowing the neuron to shift its decision threshold independently of the inputs.

The raw output of a neuron is often called an activation or pre-activation value. In many models, this value is a real number that can be large, small, positive, or negative. When neurons are organized into layers and combined with suitable activation functions (such as sigmoid, tanh, or ReLU) and appropriate normalization or output layers (such as softmax), the outputs of the network can be interpreted as scores or probabilities associated with different classes or outcomes. Thus, the role of each neuron is to transform the incoming information into a scalar value that contributes to these final probabilistic interpretations.

Three concepts are central to this behavior:

- **Bias**: A constant term added to the weighted sum of inputs. It allows the neuron to activate even when all inputs are zero and shifts the decision threshold of the neuron. In geometric terms, instead of always passing through the origin, the decision boundary can shift in the input space.

- **Activation function**: A (usually nonlinear) function applied to the weighted sum plus bias. Without nonlinearity, stacking multiple linear neurons would still produce a linear transformation overall. Activation functions such as sigmoid, tanh, and ReLU allow networks to model complex, nonlinear relationships and to build deep hierarchies of features.

- **Saturation**: A mechanism that limits the output to lie within a certain range. Conceptually, saturation prevents outputs from growing without bound. Saturation can be part of the activation function itself (as with sigmoid or tanh) or can be applied as an explicit clamping step that restricts the output to a minimum and maximum value.

A commonly used activation in modern deep learning is the **Rectified Linear Unit (ReLU)**. ReLU sets all negative inputs to zero and passes positive inputs unchanged. This simple nonlinearity helps mitigate issues like vanishing gradients, encourages sparse activations (only some neurons are active at a time), and is computationally cheap to implement. In many practical architectures, applying a ReLU after a linear transform is sufficient to build powerful deep networks.

While the conceptual description of neurons is typically expressed in terms of real numbers and floating-point arithmetic, many embedded and hardware-oriented implementations use fixed-point arithmetic to achieve efficient, deterministic, and low-power operation. In fixed-point representations, real-valued quantities are mapped to integers by assuming a fixed binary point position. For example, a real number \( r \) might be represented as an integer \( q \) such that \( r = q / 2^F \), where \( F \) is the number of fractional bits. All operations are performed on integers, and the binary point is tracked implicitly.

Fixed-point arithmetic offers several advantages in embedded systems and hardware:

- It avoids the complexity and resource usage associated with floating-point units.
- It enables predictable execution time and simpler hardware.
- It can be tuned for a particular dynamic range and precision by choosing appropriate bit widths and binary point positions.

However, fixed-point arithmetic also introduces challenges, such as saturation and quantization effects. When an intermediate result exceeds the representable range, saturation (or wraparound, depending on design choices) occurs. Quantization errors arise when mapping real numbers to discrete integer values. These effects must be carefully managed when implementing neural network computations in hardware.

In this specification, we focus on a neuron model that conceptually follows the standard weighted-sum-plus-bias pattern with an optional ReLU activation, while being amenable to a fixed-point (integer) implementation where saturation to a limited output range is explicitly applied.

---

## 2. Overview

In order to create an artificial neuron that is suitable for efficient computation, we start from the standard real-valued formulation and then adapt it to a form that can be implemented using integer or fixed-point arithmetic.

Conceptually, a neuron combines a vector of inputs with a vector of weights and a bias:

```math
y = b + \sum_{i=0}^{N-1} x_i \cdot w_i
````

Here:

* ( x_i ) are the inputs (features) for a single data instance.
* ( w_i ) are the corresponding weights that encode how important each input is.
* ( b ) is the bias term that shifts the neuron's activation threshold.
* ( y ) is the raw output (pre-activation) of the neuron.

In a purely real-valued setting, one could directly implement this computation using floating-point operations. However, in many embedded systems and hardware accelerators, it is more practical to represent ( x_i ), ( w_i ), and ( b ) as fixed-point quantities. Each real-valued variable is scaled and mapped to an integer by choosing a suitable binary point position. The computation then proceeds on these integer representations.

To align with this practice, we consider a version of the neuron where:

* The inputs, weights, and bias are treated as signed integers corresponding to fixed-point quantities.
* The sum of products is computed using integer arithmetic.
* After computing the sum, an optional ReLU activation is applied:

  * If the result is negative, it is replaced by zero.
  * If it is nonnegative, it is left unchanged.
* Finally, the result is saturated to a specified signed integer range, reflecting a finite output precision. Any value that exceeds this range is clamped to the nearest representable boundary.

This sequence of operations—weighted sum, bias addition, optional ReLU, and output saturation—captures the core behavior of a single artificial neuron in a form that can be implemented using fixed-point integer arithmetic, while still preserving the essential conceptual properties used in neural networks.

---

## 3. Mathematical Behavior

The conceptual neuron model can be described in two stages:

1. A general real-valued formulation.
2. A fixed-point integer formulation that closely mirrors the real-valued behavior under suitable scaling.

### 3.1 Real-Valued Neuron

Let:

* ( N ) be the number of inputs.
* ( x_0, x_1, \dots, x_{N-1} ) be real-valued inputs.
* ( w_0, w_1, \dots, w_{N-1} ) be real-valued weights.
* ( b ) be a real-valued bias.

The standard real-valued neuron computes the weighted sum plus bias:

```math
s = b + \sum_{i=0}^{N-1} x_i \cdot w_i
```

An activation function ( f(\cdot) ) is then applied:

```math
y_{\text{real}} = f(s)
```

For a ReLU activation, this is:

```math
y_{\text{real}} =
\begin{cases}
0, & \text{if } s < 0 \\
s, & \text{if } s \ge 0
\end{cases}
```

In a full neural network, additional normalization or output layers (such as softmax) can transform such activations into probabilities or calibrated scores. In this specification, we focus on the local behavior of a single neuron: computing ( s ) and applying a ReLU-like nonlinearity.

### 3.2 Fixed-Point Integer Neuron

To make the neuron suitable for fixed-point integer implementation, each real-valued quantity is approximated by an integer through scaling. Conceptually, we assume that there exists a fixed binary point position so that:

* ( x_i ) are represented as signed integers corresponding to scaled real inputs.
* ( w_i ) are signed integers corresponding to scaled real weights.
* ( b ) is a signed integer corresponding to a scaled real bias.

The internal computation then proceeds purely on integers. Let us denote the integer representations by the same symbols for simplicity, with the understanding that they stand for fixed-point encoded values.

1. **Initialization (integer domain)**

```math
\text{acc}_0 = \text{bias}
```

2. **Integer multiply–accumulate**

For ( k = 0, 1, \dots, N-1 ):

```math
\text{acc}_{k+1} = \text{acc}_k + x_k \cdot w_k
```

After processing all inputs:

```math
\text{acc}_{\text{final}} = \text{bias} + \sum_{i=0}^{N-1} x_i \cdot w_i
```

This mirrors the real-valued sum, but all quantities are integers that implicitly represent scaled real numbers.

3. **Optional ReLU activation (integer domain)**

We then apply an optional ReLU-like activation to the integer accumulator:

```math
\text{acc}_{\text{relu}} =
\begin{cases}
0, & \text{if } \text{acc}_{\text{final}} < 0 \\
\text{acc}_{\text{final}}, & \text{if } \text{acc}_{\text{final}} \ge 0
\end{cases}
```

If ReLU is disabled, we simply take:

```math
\text{acc}_{\text{relu}} = \text{acc}_{\text{final}}
```

4. **Saturation to a finite output range**

Because the integer representation has a finite number of bits, the output must lie within a specific signed range. Let `OUT_W` be the number of bits used for the output. The representable range is:

```math
\text{OUT\_MIN} = -2^{\text{OUT\_W}-1}
```

```math
\text{OUT\_MAX} = 2^{\text{OUT\_W}-1} - 1
```

The final integer output ( y ) is defined via saturation (clamping):

```math
y =
\begin{cases}
\text{OUT\_MAX}, & \text{if } \text{acc}_{\text{relu}} > \text{OUT\_MAX} \\
\text{OUT\_MIN}, & \text{if } \text{acc}_{\text{relu}} < \text{OUT\_MIN} \\
\text{acc}_{\text{relu}}, & \text{otherwise}
\end{cases}
```

In other words, if the activated value exceeds the representable range, it is clipped to the nearest boundary. When mapped back through the fixed-point scaling, this corresponds to a real-valued neuron whose output is limited to a certain interval.

From this point onward, we refer to this fixed-point integer formulation as the target behavior: any implementation should perform an integer multiply–accumulate, optional ReLU activation, and saturation exactly as described above (up to the fixed-point scaling convention).

---

## 4. Working Example

Consider a simple configuration with:

* Number of inputs: ( N = 2 ).
* Integer (fixed-point encoded) inputs: ( x = [10, -3] ).
* Integer (fixed-point encoded) weights: ( w = [7, 20] ).
* Integer bias: ( \text{bias} = 100 ).
* ReLU activation enabled.
* An output width `OUT_W` large enough that no saturation occurs in this example.

1. **Initialization**

```math
\text{acc}_0 = \text{bias} = 100
```

2. **First input–weight pair**

```math
10 \cdot 7 = 70
```

```math
\text{acc}_1 = \text{acc}_0 + 70 = 100 + 70 = 170
```

3. **Second input–weight pair**

```math
-3 \cdot 20 = -60
```

```math
\text{acc}_2 = \text{acc}_1 + (-60) = 170 - 60 = 110
```

Thus:

```math
\text{acc}_{\text{final}} = 110
```

4. **ReLU activation**

With ReLU enabled:

```math
\text{acc}_{\text{relu}} =
\begin{cases}
0, & \text{if } 110 < 0 \\
110, & \text{if } 110 \ge 0
\end{cases}
= 110
```

5. **Saturation**

Assume the signed output range includes 110, so there is no need to clamp:

```math
y = 110
```

Interpreted as a fixed-point value, this output corresponds to a scaled real number whose magnitude and sign reflect the neuron's response to the given inputs, weights, and bias. Any correct implementation of the fixed-point neuron must reproduce this integer result for the given configuration.
