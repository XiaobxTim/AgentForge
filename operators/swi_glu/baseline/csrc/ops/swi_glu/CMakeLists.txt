# SwiGLU Direct Launch operator

set(SWI_GLU_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/swi_glu_kernel.cpp
)

set(SWI_GLU_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/swi_glu_plugin.cpp
)

register_direct_launch_op(
    "${SWI_GLU_KERNEL_SRCS}"
    op_kernel
    "${SWI_GLU_PLUGIN_SRCS}"
    op_kernel
)
