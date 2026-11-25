# Device selection and locks

[`resource.sh`](resource.sh) provides GPU and NPU selection and locking helpers for local [`run_test.sh`](../run_test.sh) and [`run_act.sh`](../run_act.sh) runs and the [L2 resource probe](../ci/run_resource_level.sh) in CI.
On Ascend, `lock_npu` starts a background `sleep` process as a marker. The process does not reserve an NPU.
