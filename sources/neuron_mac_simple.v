`timescale 1ns/1ps
// ============================================================
// neuron_mac_simple.v  (Verilog-2001)
// Simplified serial neuron: y = bias + sum(x[i]*w[i])
// - No fractional-bit alignment or rounding
// - Optional ReLU
// - Simple saturation to OUT_W
// ============================================================
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

    function integer clog2;
        input integer value;
        integer v;
        begin
            v = value - 1;
            clog2 = 0;
            while (v > 0) begin
                clog2 = clog2 + 1;
                v = v >> 1;
            end
        end
    endfunction

    localparam integer PROD_W   = X_W + W_W;
    localparam integer SUM_GROW = (NUM_INPUTS <= 1) ? 1 : clog2(NUM_INPUTS);
    localparam integer ACC_W    = PROD_W + SUM_GROW + GUARD_BITS;
    localparam integer CNT_W    = (NUM_INPUTS <= 1) ? 1 : clog2(NUM_INPUTS);

    reg [NUM_INPUTS*X_W-1:0] x_shift;
    reg [NUM_INPUTS*W_W-1:0] w_shift;
    reg [CNT_W-1:0]          count;

    reg signed [ACC_W-1:0]   acc;

    wire signed [X_W-1:0] x_i = $signed(x_shift[X_W-1:0]);
    wire signed [W_W-1:0] w_i = $signed(w_shift[W_W-1:0]);

    wire signed [PROD_W-1:0] prod = x_i * w_i;
    wire signed [ACC_W-1:0]  acc_next = acc + $signed(prod);

    // ReLU and saturation (combinational)
    wire signed [ACC_W-1:0] relu_v = (USE_RELU != 0 && acc_next < 0) ? $signed(0) : acc_next;

    wire signed [ACC_W-1:0] out_max_acc = $signed({1'b0, {(OUT_W-1){1'b1}}});
    wire signed [ACC_W-1:0] out_min_acc = $signed({1'b1, {(OUT_W-1){1'b0}}});

    wire signed [OUT_W-1:0] sat_out =
        (relu_v > out_max_acc) ? $signed({1'b0, {(OUT_W-1){1'b1}}}) :
        (relu_v < out_min_acc) ? $signed({1'b1, {(OUT_W-1){1'b0}}}) :
                                 relu_v[OUT_W-1:0];

    assign in_ready = ~busy;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy      <= 1'b0;
            out_valid <= 1'b0;
            out_data  <= '0;
            x_shift   <= '0;
            w_shift   <= '0;
            acc       <= '0;
            count     <= '0;
        end else begin
            out_valid <= 1'b0;

            if (in_valid && in_ready) begin
                busy    <= 1'b1;
                count   <= '0;
                x_shift <= x_flat;
                w_shift <= w_flat;
                acc     <= bias;  // no scaling/alignment
            end else if (busy) begin
                acc     <= acc_next;
                x_shift <= x_shift >> X_W;
                w_shift <= w_shift >> W_W;

                if (count == (NUM_INPUTS-1)) begin
                    busy      <= 1'b0;
                    out_valid <= 1'b1;
                    out_data  <= sat_out;
                end else begin
                    count <= count + 1'b1;
                end
            end
        end
    end

endmodule
