`timescale 1ns/1ps
module neuron_mac_serial #(
    parameter integer NUM_INPUTS = 8,

    parameter integer X_W       = 8,   // input sample width (signed)
    parameter integer W_W       = 8,   // weight width (signed)
    parameter integer B_W       = 32,  // bias width (signed)
    parameter integer OUT_W     = 16,  // output width (signed)

    // Fixed-point fractional bits for each quantity.
    // x is Q?.X_FRAC, w is Q?.W_FRAC, bias is Q?.B_FRAC, output is Q?.OUT_FRAC
    parameter integer X_FRAC    = 4,
    parameter integer W_FRAC    = 4,
    parameter integer B_FRAC    = 8,
    parameter integer OUT_FRAC  = 8,

    // Extra headroom in accumulator beyond sum of products.
    parameter integer GUARD_BITS = 2,

    // 1 = apply ReLU (clamp negative sum to 0) before quantize/saturate
    parameter integer USE_RELU  = 1
)(
    input  wire                         clk,
    input  wire                         rst_n,

    // Input handshake: present x_flat/w_flat/bias with in_valid=1.
    // Module asserts in_ready when it can accept a new neuron operation.
    input  wire                         in_valid,
    output wire                         in_ready,
    input  wire signed [B_W-1:0]        bias,
    input  wire        [NUM_INPUTS*X_W-1:0] x_flat,
    input  wire        [NUM_INPUTS*W_W-1:0] w_flat,

    // Output handshake: out_valid pulses for 1 cycle with out_data.
    output reg                          out_valid,
    output reg  signed [OUT_W-1:0]      out_data,

    output reg                          busy
);

// Insert your implementation here

endmodule
