# mHC Sinkhorn Direct Launch operator

set(MHC_SINKHORN_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/mhc_sinkhorn_kernel.cpp
)

set(MHC_SINKHORN_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/mhc_sinkhorn_plugin.cpp
)

register_direct_launch_op(
    "${MHC_SINKHORN_KERNEL_SRCS}"
    op_kernel
    "${MHC_SINKHORN_PLUGIN_SRCS}"
    op_kernel
)
