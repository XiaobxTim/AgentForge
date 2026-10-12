#ifndef AGENTFORGE_MHC_SINKHORN_LAUNCH_H
#define AGENTFORGE_MHC_SINKHORN_LAUNCH_H

#include <cstdint>
#include <tuple>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

std::tuple<int64_t, int64_t>
calc_mhc_sinkhorn_tiling_params(
    int64_t batchSize);

extern "C" {

void launch_mhc_sinkhorn_kernel_float(
    GM_ADDR input,
    GM_ADDR output,
    int64_t batchSize,
    int64_t hcMult,
    int64_t iterStep,
    float eps,
    int64_t numBlocks,
    int64_t matricesPerBlock,
    void* stream);

}

#define MHC_SINKHORN_LAUNCH_FLOAT( \
    input, output, batch, hc, iter, eps, blocks, perBlock, stream) \
    launch_mhc_sinkhorn_kernel_float( \
        input, output, batch, hc, iter, eps, blocks, perBlock, stream)

#endif
