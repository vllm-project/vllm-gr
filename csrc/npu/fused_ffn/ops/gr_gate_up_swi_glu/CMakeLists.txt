add_ops_compile_options(OP_NAME GrGateUpSwiGlu OPTIONS
    --cce-auto-sync=off -Wno-deprecated-declarations -Wno-error
    -I${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    -mllvm -cce-aicore-stack-size=0x8000
    -mllvm -cce-aicore-function-stack-size=0x8000
    -mllvm -cce-aicore-addr-transform)
foreach(target op_host_aclnn optiling opsproto)
    target_sources(${target} PRIVATE op_host/gr_gate_up_swi_glu_cpu.cpp)
endforeach()
install(FILES op_kernel/gr_gate_up_swi_glu.cpp DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
