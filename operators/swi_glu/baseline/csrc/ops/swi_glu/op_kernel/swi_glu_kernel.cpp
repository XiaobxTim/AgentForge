#include <algorithm>
#include <tuple>
#include <type_traits>

#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

constexpr uint32_t SWI_GLU_DEFAULT_TILE_ELEMENTS = 4096;
constexpr int64_t SWI_GLU_MIN_ELEMS_PER_CORE = 4096;

template <typename T>
__aicore__ inline void ComputeSwiGluTile(
    const AscendC::LocalTensor<T>& x0Local,
    const AscendC::LocalTensor<T>& x1Local,
    const AscendC::LocalTensor<T>& outputLocal,
    const AscendC::LocalTensor<float>& x0Float,
    const AscendC::LocalTensor<float>& x1Float,
    const AscendC::LocalTensor<float>& siluFloat,
    const AscendC::LocalTensor<float>& outputFloat,
    uint32_t count)
{
    if constexpr (std::is_same<T, float>::value) {
        AscendC::Silu(siluFloat, x0Local, count);
        AscendC::Mul(outputLocal, siluFloat, x1Local, count);
    } else {
        // The CANN Bench golden computes FP16/BF16 inputs in FP32.
        AscendC::Cast(
            x0Float,
            x0Local,
            AscendC::RoundMode::CAST_NONE,
            count);

        AscendC::Cast(
            x1Float,
            x1Local,
            AscendC::RoundMode::CAST_NONE,
            count);

        AscendC::Silu(
            siluFloat,
            x0Float,
            count);

        AscendC::Mul(
            outputFloat,
            siluFloat,
            x1Float,
            count);

        AscendC::Cast(
            outputLocal,
            outputFloat,
            AscendC::RoundMode::CAST_RINT,
            count);
    }
}

template <typename T>
__global__ __aicore__ void swi_glu_kernel(
    GM_ADDR input,
    GM_ADDR output,
    int64_t totalOutputLength,
    int64_t groupLength,
    int64_t blockLength,
    uint32_t tileElements)
{
    AscendC::TPipe pipe;

    AscendC::GlobalTensor<T> inputGm;
    AscendC::GlobalTensor<T> outputGm;

    inputGm.SetGlobalBuffer((__gm__ T*)input);
    outputGm.SetGlobalBuffer((__gm__ T*)output);

    AscendC::TQue<AscendC::QuePosition::VECIN, 1> x0Queue;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> x1Queue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outputQueue;

    pipe.InitBuffer(
        x0Queue,
        1,
        tileElements * sizeof(T));

    pipe.InitBuffer(
        x1Queue,
        1,
        tileElements * sizeof(T));

    pipe.InitBuffer(
        outputQueue,
        1,
        tileElements * sizeof(T));

    AscendC::TBuf<AscendC::TPosition::VECCALC> x0FloatBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> x1FloatBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> siluFloatBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputFloatBuf;

    pipe.InitBuffer(
        x0FloatBuf,
        tileElements * sizeof(float));

    pipe.InitBuffer(
        x1FloatBuf,
        tileElements * sizeof(float));

    pipe.InitBuffer(
        siluFloatBuf,
        tileElements * sizeof(float));

    pipe.InitBuffer(
        outputFloatBuf,
        tileElements * sizeof(float));

    auto x0Float = x0FloatBuf.Get<float>();
    auto x1Float = x1FloatBuf.Get<float>();
    auto siluFloat = siluFloatBuf.Get<float>();
    auto outputFloat = outputFloatBuf.Get<float>();

    const int64_t blockId = AscendC::GetBlockIdx();
    const int64_t blockStart = blockId * blockLength;

    if (blockStart >= totalOutputLength) {
        return;
    }

    int64_t currentBlockLength =
        totalOutputLength - blockStart;

    if (currentBlockLength > blockLength) {
        currentBlockLength = blockLength;
    }

    int64_t processed = 0;

    while (processed < currentBlockLength) {
        const int64_t outputIndex =
            blockStart + processed;

        const int64_t offsetInGroup =
            outputIndex % groupLength;

        const int64_t remainInGroup =
            groupLength - offsetInGroup;

        int64_t count =
            currentBlockLength - processed;

        if (count > remainInGroup) {
            count = remainInGroup;
        }

        if (count > static_cast<int64_t>(tileElements)) {
            count = tileElements;
        }

        const int64_t groupId =
            outputIndex / groupLength;

        const int64_t x0Index =
            groupId * (2 * groupLength)
            + offsetInGroup;

        const int64_t x1Index =
            x0Index + groupLength;

        AscendC::DataCopyExtParams copyParams{
            1,
            static_cast<uint32_t>(count * sizeof(T)),
            0,
            0,
            0
        };

        AscendC::DataCopyPadExtParams<T> padParams{
            false,
            0,
            0,
            0
        };

        auto x0Local =
            x0Queue.AllocTensor<T>();

        auto x1Local =
            x1Queue.AllocTensor<T>();

        AscendC::DataCopyPad(
            x0Local,
            inputGm[x0Index],
            copyParams,
            padParams);

        AscendC::DataCopyPad(
            x1Local,
            inputGm[x1Index],
            copyParams,
            padParams);

        x0Queue.EnQue(x0Local);
        x1Queue.EnQue(x1Local);

        x0Local =
            x0Queue.DeQue<T>();

        x1Local =
            x1Queue.DeQue<T>();

        auto outputLocal =
            outputQueue.AllocTensor<T>();

        ComputeSwiGluTile<T>(
            x0Local,
            x1Local,
            outputLocal,
            x0Float,
            x1Float,
            siluFloat,
            outputFloat,
            static_cast<uint32_t>(count));

        outputQueue.EnQue(outputLocal);

        x0Queue.FreeTensor(x0Local);
        x1Queue.FreeTensor(x1Local);

        outputLocal =
            outputQueue.DeQue<T>();

        AscendC::DataCopyPad(
            outputGm[outputIndex],
            outputLocal,
            copyParams);

        outputQueue.FreeTensor(outputLocal);

        processed += count;
    }
}

std::tuple<int64_t, int64_t, uint32_t>
calc_swi_glu_tiling_params(
    int64_t totalOutputLength)
{
    auto platform =
        platform_ascendc::PlatformAscendCManager::GetInstance();

    int64_t coreNum =
        platform->GetCoreNumAiv();

    if (coreNum <= 0) {
        coreNum = 1;
    }

    int64_t numBlocks =
        (totalOutputLength
         + SWI_GLU_MIN_ELEMS_PER_CORE - 1)
        / SWI_GLU_MIN_ELEMS_PER_CORE;

    numBlocks =
        std::min(coreNum, numBlocks);

    if (numBlocks <= 0) {
        numBlocks = 1;
    }

    const int64_t blockLength =
        (totalOutputLength + numBlocks - 1)
        / numBlocks;

    return std::make_tuple(
        numBlocks,
        blockLength,
        SWI_GLU_DEFAULT_TILE_ELEMENTS);
}

extern "C" {

void launch_swi_glu_kernel_float(
    GM_ADDR input,
    GM_ADDR output,
    int64_t totalOutputLength,
    int64_t groupLength,
    int64_t numBlocks,
    int64_t blockLength,
    uint32_t tileElements,
    void* stream)
{
    swi_glu_kernel<float>
        <<<numBlocks, nullptr, stream>>>(
            input,
            output,
            totalOutputLength,
            groupLength,
            blockLength,
            tileElements);
}

void launch_swi_glu_kernel_half(
    GM_ADDR input,
    GM_ADDR output,
    int64_t totalOutputLength,
    int64_t groupLength,
    int64_t numBlocks,
    int64_t blockLength,
    uint32_t tileElements,
    void* stream)
{
    swi_glu_kernel<half>
        <<<numBlocks, nullptr, stream>>>(
            input,
            output,
            totalOutputLength,
            groupLength,
            blockLength,
            tileElements);
}

void launch_swi_glu_kernel_bfloat16(
    GM_ADDR input,
    GM_ADDR output,
    int64_t totalOutputLength,
    int64_t groupLength,
    int64_t numBlocks,
    int64_t blockLength,
    uint32_t tileElements,
    void* stream)
{
    swi_glu_kernel<bfloat16_t>
        <<<numBlocks, nullptr, stream>>>(
            input,
            output,
            totalOutputLength,
            groupLength,
            blockLength,
            tileElements);
}

}
