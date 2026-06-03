#!/bin/bash
LOG=/home/xl434/vla-to-npu/test_results.log
echo "===== TEST RUN: $(date) =====" > $LOG

run_test() {
    local label="$1"
    local dir="$2"
    local script="$3"
    echo "" >> $LOG
    echo "---------- $label ----------" >> $LOG
    echo "$ cd $dir && python $script" >> $LOG
    cd /home/xl434/vla-to-npu/$dir
    python $script >> $LOG 2>&1
    if [ $? -eq 0 ]; then
        echo ">>> PASSED" >> $LOG
    else
        echo ">>> FAILED" >> $LOG
    fi
}

# Kernel-level tests
run_test "test_rope_full (expected FAIL)" "kernel_testing/rope"  "test_rope_full.py"
run_test "test_sin_cos"                   "kernel_testing/rope"  "test_sin_cos.py"
run_test "test_rope_fused"                "kernel_testing/rope"  "test_rope_fused.py"
run_test "test_rope_multiple"             "kernel_testing/rope"  "test_rope_multiple.py"
run_test "test_gelu_bf16_new"             "kernel_testing/gelu"  "test_gelu_bf16_new.py"
run_test "test_silu_bf16_new"             "kernel_testing/silu"  "test_silu_bf16_new.py"

# VLA component tests
run_test "text_encoder_bf16"     "vla"  "text_encoder_bf16.py"
run_test "action_expert_bf16"    "vla"  "action_expert_bf16.py"
run_test "llama_block_rope_bf16" "vla"  "llama_block_rope_bf16.py"

# Full end-to-end
run_test "vla_e2e"               "vla"  "vla.py"

echo "" >> $LOG
echo "===== DONE: $(date) =====" >> $LOG
