#ifndef AGENTFORGE_ENGRAM_GATE_FUSION_LAUNCH_H
#define AGENTFORGE_ENGRAM_GATE_FUSION_LAUNCH_H

#include <cstdint>
#include <tuple>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

std::tuple<int64_t, int64_t>
calc_engram_stage_a_tiling_params(
    int64_t totalItems);

extern "C" {

void launch_engram_stage_a(
    GM_ADDR keys,
    GM_ADDR hiddenStates,
    GM_ADDR value,
    GM_ADDR norm1Weight,
    GM_ADDR norm2Weight,
    GM_ADDR convNormWeight,
    GM_ADDR valueGated,
    GM_ADDR normedValueGated,
    int64_t batchSize,
    int64_t seqLength,
    int64_t hcMult,
    int64_t hiddenSize,
    float normEps,
    int64_t numBlocks,
    int64_t itemsPerBlock,
    void* stream);

}

#define ENGRAM_STAGE_A_LAUNCH( \
    keys, hidden, value, w1, w2, convw, \
    vg, nvg, B, L, HC, D, eps, blocks, perBlock, stream) \
    launch_engram_stage_a( \
        keys, hidden, value, w1, w2, convw, \
        vg, nvg, B, L, HC, D, eps, blocks, perBlock, stream)

#endif
