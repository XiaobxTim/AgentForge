#ifndef AGENTFORGE_SWI_GLU_LAUNCH_H
#define AGENTFORGE_SWI_GLU_LAUNCH_H

#include <cstdint>
#include <tuple>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

std::tuple<int64_t, int64_t, uint32_t>
calc_swi_glu_tiling_params(int64_t totalOutputLength);

extern "C" {

void launch_swi_glu_kernel_float(
    GM_ADDR input,
    GM_ADDR output,
    int64_t totalOutputLength,
    int64_t groupLength,
    int64_t numBlocks,
    int64_t blockLength,
    uint32_t tileElements,
    void* stream);

void launch_swi_glu_kernel_half(
    GM_ADDR input,
    GM_ADDR output,
    int64_t totalOutputLength,
    int64_t groupLength,
    int64_t numBlocks,
    int64_t blockLength,
    uint32_t tileElements,
    void* stream);

void launch_swi_glu_kernel_bfloat16(
    GM_ADDR input,
    GM_ADDR output,
    int64_t totalOutputLength,
    int64_t groupLength,
    int64_t numBlocks,
    int64_t blockLength,
    uint32_t tileElements,
    void* stream);

}

#define SWI_GLU_LAUNCH_FLOAT(input, output, total, group, blocks, blockLen, tile, stream) \
    launch_swi_glu_kernel_float(input, output, total, group, blocks, blockLen, tile, stream)

#define SWI_GLU_LAUNCH_HALF(input, output, total, group, blocks, blockLen, tile, stream) \
    launch_swi_glu_kernel_half(input, output, total, group, blocks, blockLen, tile, stream)

#define SWI_GLU_LAUNCH_BFLOAT16(input, output, total, group, blocks, blockLen, tile, stream) \
    launch_swi_glu_kernel_bfloat16(input, output, total, group, blocks, blockLen, tile, stream)

#endif
