"""OpenOneRec benchmark helpers.

Patches are applied explicitly by entrypoints (one_rec_main.py / one_rec_acc_test.py)
to avoid polluting vllm globals on every package import. See one_rec_patch.py.
"""
