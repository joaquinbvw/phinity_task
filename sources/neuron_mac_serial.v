`timescale 1ns/1ps
// ============================================================
// neuron_mac_serial.v  (Verilog-2001)
// Serial dot-product neuron: y = sum(x[i]*w[i]) + bias
// Fixed-point with rounding + saturation, optional ReLU.
// ============================================================

module neuron_mac_serial #(
    parameter integer NUM_INPUTS = 8,

    parameter integer X_W       = 8,   // input sample width (signed)
    parameter integer W_W       = 8,   // weight width (signed)
    parameter integer B_W       = 32,  // bias width (signed)
    parameter integer OUT_W     = 16,  // output width (signed)

    // Fixed-point fractional bits for each quantity.
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

    // -------------------------
    // Small utility: clog2
    // -------------------------
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

    localparam integer PROD_W  = X_W + W_W;
    localparam integer FRAC_P  = X_FRAC + W_FRAC;

    // Accumulator width: product width + log2(NUM_INPUTS) + guard bits
    localparam integer SUM_GROW = (NUM_INPUTS <= 1) ? 1 : clog2(NUM_INPUTS);
    localparam integer ACC_W    = PROD_W + SUM_GROW + GUARD_BITS;

    // Counter/index width (at least 1)
    localparam integer CNT_W = (NUM_INPUTS <= 1) ? 1 : clog2(NUM_INPUTS);

    // -------------------------
    // Saturate ACC_W -> OUT_W
    // -------------------------
    function signed [OUT_W-1:0] sat_to_out;
        input signed [ACC_W-1:0] v;
        reg signed [ACC_W-1:0] max_v;
        reg signed [ACC_W-1:0] min_v;
        begin
            // Max:  0x7FF.. , Min: 0x800..
            max_v = $signed({1'b0, {(OUT_W-1){1'b1}}});
            min_v = $signed({1'b1, {(OUT_W-1){1'b0}}});

            if (v > max_v)       sat_to_out = {1'b0, {(OUT_W-1){1'b1}}};
            else if (v < min_v)  sat_to_out = {1'b1, {(OUT_W-1){1'b0}}};
            else                 sat_to_out = v[OUT_W-1:0];
        end
    endfunction

    // -------------------------
    // Align bias into ACC scale (FRAC_P fractional bits)
    // Includes rounding when shifting right.
    //
    // NOTE: If B_W > ACC_W, the assignment truncates to ACC_W bits
    // (this is intentional and is mirrored in the Python model).
    // -------------------------
    function signed [ACC_W-1:0] align_bias;
        input signed [B_W-1:0] b;
        reg   signed [ACC_W-1:0] be;
        reg   signed [ACC_W-1:0] round_const;
        integer sh;
        begin
            // Truncate or sign-extend into ACC_W
            be = b;

            if (B_FRAC > FRAC_P) begin
                sh = (B_FRAC - FRAC_P);          // right shift amount (>=1)
                round_const = $signed(1) <<< (sh-1);
                if (be >= 0) be = be + round_const;
                else         be = be - round_const;
                align_bias = be >>> sh;
            end else if (FRAC_P > B_FRAC) begin
                sh = (FRAC_P - B_FRAC);
                align_bias = be <<< sh;
            end else begin
                align_bias = be;
            end
        end
    endfunction

    // -------------------------
    // Quantize ACC scale (FRAC_P) -> OUT scale (OUT_FRAC),
    // then saturate to OUT_W. Rounds on right shifts.
    // -------------------------
    function signed [OUT_W-1:0] quantize_and_sat;
        input signed [ACC_W-1:0] vin;
        reg   signed [ACC_W-1:0] v;
        reg   signed [ACC_W-1:0] round_const;
        integer sh;
        begin
            v = vin;

            // Optional ReLU
            if (USE_RELU != 0) begin
                if (v < 0) v = 0;
            end

            // Adjust fractional bits from FRAC_P to OUT_FRAC
            if (FRAC_P > OUT_FRAC) begin
                sh = (FRAC_P - OUT_FRAC);       // right shift amount (>=1)
                round_const = $signed(1) <<< (sh-1);
                if (v >= 0) v = v + round_const;
                else        v = v - round_const;
                v = v >>> sh;                   // arithmetic shift
            end else if (OUT_FRAC > FRAC_P) begin
                sh = (OUT_FRAC - FRAC_P);
                v = v <<< sh;
            end

            quantize_and_sat = sat_to_out(v);
        end
    endfunction

    // -------------------------
    // Serial MAC datapath (indexed, not shifting)
    // -------------------------

    // Latched input vectors
    reg [NUM_INPUTS*X_W-1:0] x_reg;
    reg [NUM_INPUTS*W_W-1:0] w_reg;

    // Accumulator and element index
    reg signed [ACC_W-1:0]   acc;
    reg [CNT_W-1:0]          idx;

    // Current element (little-endian packing by index)
    wire signed [X_W-1:0] x_i;
    wire signed [W_W-1:0] w_i;

    // Dynamic part-select: x[i] and w[i]
    assign x_i = $signed(x_reg[idx*X_W +: X_W]);
    assign w_i = $signed(w_reg[idx*W_W +: W_W]);

    // Product and extended version in accumulator width
    wire signed [PROD_W-1:0] prod     = x_i * w_i;
    wire signed [ACC_W-1:0]  prod_acc = prod;      // sign-extend
    wire signed [ACC_W-1:0]  acc_next = acc + prod_acc;

    // Ready is simply the complement of busy (combinational)
    assign in_ready = ~busy;

    // -------------------------
    // Main sequential logic
    // -------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy      <= 1'b0;
            out_valid <= 1'b0;
            out_data  <= {OUT_W{1'b0}};
            x_reg     <= {NUM_INPUTS*X_W{1'b0}};
            w_reg     <= {NUM_INPUTS*W_W{1'b0}};
            acc       <= {ACC_W{1'b0}};
            idx       <= {CNT_W{1'b0}};
        end else begin
            // out_valid is a one-cycle pulse
            out_valid <= 1'b0;

            // Accept a new operation only when idle and in_valid is high
            if (in_valid && in_ready) begin
                busy  <= 1'b1;
                idx   <= {CNT_W{1'b0}};

                // Latch full input vectors (element 0 in LSBs)
                x_reg <= x_flat;
                w_reg <= w_flat;

                // Initialize accumulator with aligned bias (FRAC_P scale)
                acc   <= align_bias(bias);
            end
            else if (busy) begin
                // Consume one x/w pair per cycle in index order
                acc <= acc_next;

                if (idx == (NUM_INPUTS-1)) begin
                    // Last element: produce result and return to idle
                    busy      <= 1'b0;
                    out_valid <= 1'b1;
                    out_data  <= quantize_and_sat(acc_next);
                end else begin
                    idx <= idx + 1'b1;
                end
            end
        end
    end

endmodule
