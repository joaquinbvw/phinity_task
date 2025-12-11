`timescale 1ns/1ps
module neuron_mac_simple #(
    parameter integer NUM_INPUTS = 8,
    parameter integer X_W        = 8,   // signed
    parameter integer W_W        = 8,   // signed
    parameter integer B_W        = 16,  // signed
    parameter integer OUT_W      = 16,  // signed
    parameter integer GUARD_BITS = 2,
    parameter integer USE_RELU   = 1
)(
    input  wire                         clk,
    input  wire                         rst_n,

    input  wire                         in_valid,
    output wire                         in_ready,
    input  wire signed [B_W-1:0]        bias,
    input  wire        [NUM_INPUTS*X_W-1:0] x_flat,
    input  wire        [NUM_INPUTS*W_W-1:0] w_flat,

    output reg                          out_valid,
    output reg  signed [OUT_W-1:0]      out_data,
    output reg                          busy
);

//Insert utility function that computes the clog2() for positive integers. Must match the behavior required by the spec and be usable for scaling the width of internal signals

//Define derived widths / localparams and internal signal declarations

//Insert the current element extraction, multiply, and accumulate extracting the inputs and weights from LSB slices of shift registers, interpret them as signed, compute prod and acc_next correctly (combinational part).

//Compute the ReLU + saturation (combinational) part.

//Compute handshake for the capture of the signals.

//Insert the main sequential control (reset, accept, process, finish) on one single always block, you should capture the inputs on this part and start the computation of the multiply/accumulate operations, one single input and weight per clock cycle according to the specs.

endmodule
