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

std::tuple<int64_t, int64_t>
calc_engram_stage_b_tiling_params(
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

void launch_engram_stage_b_prefill(
    GM_ADDR valueGated,
    GM_ADDR normedValueGated,
    GM_ADDR convWeight,
    GM_ADDR output,
    GM_ADDR convStateOut,
    int64_t batchSize,
    int64_t seqLength,
    int64_t hcMult,
    int64_t hiddenSize,
    int64_t kernelSize,
    int64_t dilation,
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

#define ENGRAM_STAGE_B_PREFILL_LAUNCH( \
    vg, nvg, convw, output, stateOut, \
    B, L, HC, D, K, dil, blocks, perBlock, stream) \
    launch_engram_stage_b_prefill( \
        vg, nvg, convw, output, stateOut, \
        B, L, HC, D, K, dil, blocks, perBlock, stream)


extern "C" {

void launch_engram_stage_b_decode(
    GM_ADDR valueGated,
    GM_ADDR normedValueGated,
    GM_ADDR convWeight,
    GM_ADDR convState,
    GM_ADDR output,
    GM_ADDR convStateOut,
    int64_t batchSize,
    int64_t seqLength,
    int64_t hcMult,
    int64_t hiddenSize,
    int64_t kernelSize,
    int64_t dilation,
    int64_t numBlocks,
    int64_t itemsPerBlock,
    void* stream);

}

#define ENGRAM_STAGE_B_DECODE_LAUNCH( \
    vg, nvg, convw, state, output, stateOut, \
    B, L, HC, D, K, dil, blocks, perBlock, stream) \
    launch_engram_stage_b_decode( \
        vg, nvg, convw, state, output, stateOut, \
        B, L, HC, D, K, dil, blocks, perBlock, stream)

#endif
