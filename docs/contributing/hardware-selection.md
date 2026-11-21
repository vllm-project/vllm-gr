# Selecting test hardware

Run `python -m pytest tests --test-hardware=cuda:H100 -m 'l1 or l2'` to include
host and compatible accelerator tests. Use `host` or `npu:A3` for those targets.
CUDA supports L20, A100, H800, H100, L4 and B200. Ascend supports A2, A3 and A5.
Keep the lowercase `l4` level distinct from the uppercase `L4` GPU SKU.

Declare new cases with `@hardware_test(res={"cuda": ["H100"]}, num_cards=2)`
from `tests.helpers.hardware`. Use `hardware_marks` on individual parameter sets.
Collection checks declarations. Execution verifies the visible SKU and minimum
card count. Existing legacy markers remain supported.
