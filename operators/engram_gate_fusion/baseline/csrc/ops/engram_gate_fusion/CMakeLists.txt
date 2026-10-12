# Engram Gate Fusion Direct Launch operator

set(ENGRAM_GATE_FUSION_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/engram_gate_fusion_kernel.cpp
)

set(ENGRAM_GATE_FUSION_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/engram_gate_fusion_plugin.cpp
)

register_direct_launch_op(
    "${ENGRAM_GATE_FUSION_KERNEL_SRCS}"
    op_kernel
    "${ENGRAM_GATE_FUSION_PLUGIN_SRCS}"
    op_kernel
)
