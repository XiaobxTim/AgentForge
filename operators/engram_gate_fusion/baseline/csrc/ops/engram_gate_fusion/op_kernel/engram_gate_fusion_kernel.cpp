#include <algorithm>
#include <tuple>

#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

class EngramStageAKernel {
public:
    __aicore__ inline void Init(
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
        int64_t itemsPerBlock)
    {
        batchSize_ = batchSize;
        seqLength_ = seqLength;
        hcMult_ = hcMult;
        hiddenSize_ = hiddenSize;
        normEps_ = normEps;
        itemsPerBlock_ = itemsPerBlock;

        totalItems_ =
            batchSize_ * seqLength_ * hcMult_;

        keysGm_.SetGlobalBuffer(
            (__gm__ bfloat16_t*)keys);

        hiddenGm_.SetGlobalBuffer(
            (__gm__ bfloat16_t*)hiddenStates);

        valueGm_.SetGlobalBuffer(
            (__gm__ bfloat16_t*)value);

        norm1WeightGm_.SetGlobalBuffer(
            (__gm__ float*)norm1Weight);

        norm2WeightGm_.SetGlobalBuffer(
            (__gm__ float*)norm2Weight);

        convNormWeightGm_.SetGlobalBuffer(
            (__gm__ float*)convNormWeight);

        valueGatedGm_.SetGlobalBuffer(
            (__gm__ float*)valueGated);

        normedValueGatedGm_.SetGlobalBuffer(
            (__gm__ float*)normedValueGated);

        const uint32_t bf16Bytes =
            static_cast<uint32_t>(
                hiddenSize_ * sizeof(bfloat16_t));

        const uint32_t floatBytes =
            static_cast<uint32_t>(
                hiddenSize_ * sizeof(float));

        pipe_.InitBuffer(
            keysQueue_,
            1,
            bf16Bytes);

        pipe_.InitBuffer(
            hiddenQueue_,
            1,
            bf16Bytes);

        pipe_.InitBuffer(
            valueQueue_,
            1,
            bf16Bytes);

        pipe_.InitBuffer(
            norm1Queue_,
            1,
            floatBytes);

        pipe_.InitBuffer(
            norm2Queue_,
            1,
            floatBytes);

        pipe_.InitBuffer(
            convNormQueue_,
            1,
            floatBytes);

        pipe_.InitBuffer(
            valueGatedOutQueue_,
            1,
            floatBytes);

        pipe_.InitBuffer(
            normedValueGatedOutQueue_,
            1,
            floatBytes);

        pipe_.InitBuffer(
            keysFloatBuf_,
            floatBytes);

        pipe_.InitBuffer(
            hiddenFloatBuf_,
            floatBytes);

        pipe_.InitBuffer(
            valueFloatBuf_,
            floatBytes);

        pipe_.InitBuffer(
            temp1Buf_,
            floatBytes);

        pipe_.InitBuffer(
            temp2Buf_,
            floatBytes);

        pipe_.InitBuffer(
            valueGatedBuf_,
            floatBytes);

        pipe_.InitBuffer(
            normedValueGatedBuf_,
            floatBytes);

        pipe_.InitBuffer(
            scalarInBuf_,
            32);

        pipe_.InitBuffer(
            scalarOutBuf_,
            32);
    }

    __aicore__ inline void Process()
    {
        const int64_t blockId =
            static_cast<int64_t>(
                AscendC::GetBlockIdx());

        const int64_t firstItem =
            blockId * itemsPerBlock_;

        if (firstItem >= totalItems_) {
            return;
        }

        int64_t lastItem =
            firstItem + itemsPerBlock_;

        if (lastItem > totalItems_) {
            lastItem = totalItems_;
        }

        for (int64_t item = firstItem;
             item < lastItem;
             ++item) {
            ProcessOneItem(item);
        }
    }

private:
    __aicore__ inline void SyncVectorToScalar()
    {
        AscendC::SetFlag<
            AscendC::HardEvent::V_S>(1);

        AscendC::WaitFlag<
            AscendC::HardEvent::V_S>(1);
    }

    __aicore__ inline void SyncScalarToVector()
    {
        AscendC::SetFlag<
            AscendC::HardEvent::S_V>(0);

        AscendC::WaitFlag<
            AscendC::HardEvent::S_V>(0);
    }

    __aicore__ inline float ScalarSqrt(
        float value)
    {
        auto input =
            scalarInBuf_.Get<float>();

        auto output =
            scalarOutBuf_.Get<float>();

        input.SetValue(0, value);

        SyncScalarToVector();

        AscendC::Sqrt(
            output,
            input,
            static_cast<int32_t>(1));

        SyncVectorToScalar();

        return output.GetValue(0);
    }

    __aicore__ inline float ScalarExp(
        float value)
    {
        auto input =
            scalarInBuf_.Get<float>();

        auto output =
            scalarOutBuf_.Get<float>();

        input.SetValue(0, value);

        SyncScalarToVector();

        AscendC::Exp(
            output,
            input,
            static_cast<int32_t>(1));

        SyncVectorToScalar();

        return output.GetValue(0);
    }

    __aicore__ inline float SumLocal(
        const AscendC::LocalTensor<float>& tensor)
    {
        float sum = 0.0f;

        for (int64_t i = 0;
             i < hiddenSize_;
             ++i) {
            sum += tensor.GetValue(
                static_cast<uint32_t>(i));
        }

        return sum;
    }

    __aicore__ inline void ProcessOneItem(
        int64_t item)
    {
        const int64_t hc =
            item % hcMult_;

        const int64_t token =
            item / hcMult_;

        const int64_t activationOffset =
            item * hiddenSize_;

        const int64_t valueOffset =
            token * hiddenSize_;

        const int64_t weightOffset =
            hc * hiddenSize_;

        const uint32_t bf16Bytes =
            static_cast<uint32_t>(
                hiddenSize_
                * sizeof(bfloat16_t));

        const uint32_t floatBytes =
            static_cast<uint32_t>(
                hiddenSize_
                * sizeof(float));

        AscendC::DataCopyExtParams
            bf16CopyParams{
                1,
                bf16Bytes,
                0,
                0,
                0
            };

        AscendC::DataCopyExtParams
            floatCopyParams{
                1,
                floatBytes,
                0,
                0,
                0
            };

        AscendC::DataCopyPadExtParams<
            bfloat16_t> bf16Pad{
                false,
                0,
                0,
                0
            };

        AscendC::DataCopyPadExtParams<
            float> floatPad{
                false,
                0,
                0,
                0
            };

        auto keysLocal =
            keysQueue_
                .AllocTensor<bfloat16_t>();

        auto hiddenLocal =
            hiddenQueue_
                .AllocTensor<bfloat16_t>();

        auto valueLocal =
            valueQueue_
                .AllocTensor<bfloat16_t>();

        auto norm1Local =
            norm1Queue_
                .AllocTensor<float>();

        auto norm2Local =
            norm2Queue_
                .AllocTensor<float>();

        auto convNormLocal =
            convNormQueue_
                .AllocTensor<float>();

        AscendC::DataCopyPad(
            keysLocal,
            keysGm_[activationOffset],
            bf16CopyParams,
            bf16Pad);

        AscendC::DataCopyPad(
            hiddenLocal,
            hiddenGm_[activationOffset],
            bf16CopyParams,
            bf16Pad);

        AscendC::DataCopyPad(
            valueLocal,
            valueGm_[valueOffset],
            bf16CopyParams,
            bf16Pad);

        AscendC::DataCopyPad(
            norm1Local,
            norm1WeightGm_[weightOffset],
            floatCopyParams,
            floatPad);

        AscendC::DataCopyPad(
            norm2Local,
            norm2WeightGm_[weightOffset],
            floatCopyParams,
            floatPad);

        AscendC::DataCopyPad(
            convNormLocal,
            convNormWeightGm_[weightOffset],
            floatCopyParams,
            floatPad);

        keysQueue_.EnQue(keysLocal);
        hiddenQueue_.EnQue(hiddenLocal);
        valueQueue_.EnQue(valueLocal);
        norm1Queue_.EnQue(norm1Local);
        norm2Queue_.EnQue(norm2Local);
        convNormQueue_.EnQue(convNormLocal);

        keysLocal =
            keysQueue_
                .DeQue<bfloat16_t>();

        hiddenLocal =
            hiddenQueue_
                .DeQue<bfloat16_t>();

        valueLocal =
            valueQueue_
                .DeQue<bfloat16_t>();

        norm1Local =
            norm1Queue_
                .DeQue<float>();

        norm2Local =
            norm2Queue_
                .DeQue<float>();

        convNormLocal =
            convNormQueue_
                .DeQue<float>();

        auto keysFloat =
            keysFloatBuf_.Get<float>();

        auto hiddenFloat =
            hiddenFloatBuf_.Get<float>();

        auto valueFloat =
            valueFloatBuf_.Get<float>();

        auto temp1 =
            temp1Buf_.Get<float>();

        auto temp2 =
            temp2Buf_.Get<float>();

        auto valueGated =
            valueGatedBuf_.Get<float>();

        auto normedValueGated =
            normedValueGatedBuf_.Get<float>();

        const int32_t count =
            static_cast<int32_t>(
                hiddenSize_);

        // -------------------------------------------------
        // BF16 -> FP32
        // -------------------------------------------------

        AscendC::Cast(
            keysFloat,
            keysLocal,
            AscendC::RoundMode::CAST_NONE,
            count);

        AscendC::Cast(
            hiddenFloat,
            hiddenLocal,
            AscendC::RoundMode::CAST_NONE,
            count);

        AscendC::Cast(
            valueFloat,
            valueLocal,
            AscendC::RoundMode::CAST_NONE,
            count);

        // -------------------------------------------------
        // RMS statistics for keys
        // -------------------------------------------------

        AscendC::Mul(
            temp1,
            keysFloat,
            keysFloat,
            count);

        SyncVectorToScalar();

        const float keysSquareSum =
            SumLocal(temp1);

        const float keysMeanSquare =
            keysSquareSum
            / static_cast<float>(
                hiddenSize_);

        const float keysRms =
            ScalarSqrt(
                keysMeanSquare
                + normEps_);

        const float invKeysRms =
            1.0f / keysRms;

        // -------------------------------------------------
        // RMS statistics for hidden_states
        // -------------------------------------------------

        AscendC::Mul(
            temp1,
            hiddenFloat,
            hiddenFloat,
            count);

        SyncVectorToScalar();

        const float hiddenSquareSum =
            SumLocal(temp1);

        const float hiddenMeanSquare =
            hiddenSquareSum
            / static_cast<float>(
                hiddenSize_);

        const float hiddenRms =
            ScalarSqrt(
                hiddenMeanSquare
                + normEps_);

        const float invHiddenRms =
            1.0f / hiddenRms;

        // -------------------------------------------------
        // normed_keys
        // -------------------------------------------------

        AscendC::Muls(
            temp1,
            keysFloat,
            invKeysRms,
            count);

        AscendC::Mul(
            temp1,
            temp1,
            norm1Local,
            count);

        // -------------------------------------------------
        // normed_hidden
        // -------------------------------------------------

        AscendC::Muls(
            temp2,
            hiddenFloat,
            invHiddenRms,
            count);

        AscendC::Mul(
            temp2,
            temp2,
            norm2Local,
            count);

        // -------------------------------------------------
        // scaled dot product gate
        // -------------------------------------------------

        AscendC::Mul(
            temp1,
            temp1,
            temp2,
            count);

        SyncVectorToScalar();

        float rawGate =
            SumLocal(temp1);

        const float sqrtD =
            ScalarSqrt(
                static_cast<float>(
                    hiddenSize_));

        rawGate /= sqrtD;

        // -------------------------------------------------
        // sqrt(clamp(abs(raw_gate), 1e-6))
        // * sign(raw_gate)
        // -------------------------------------------------

        float absGate =
            rawGate >= 0.0f
                ? rawGate
                : -rawGate;

        if (absGate < 1.0e-6f) {
            absGate = 1.0e-6f;
        }

        float transformed =
            ScalarSqrt(absGate);

        if (rawGate < 0.0f) {
            transformed =
                -transformed;
        } else if (rawGate == 0.0f) {
            transformed =
                0.0f;
        }

        // sigmoid(x) = 1 / (1 + exp(-x))
        const float expValue =
            ScalarExp(-transformed);

        const float gate =
            1.0f
            / (1.0f + expValue);

        // -------------------------------------------------
        // value_gated
        // -------------------------------------------------

        AscendC::Muls(
            valueGated,
            valueFloat,
            gate,
            count);

        // -------------------------------------------------
        // RMSNorm(value_gated, conv_norm_weight)
        // -------------------------------------------------

        AscendC::Mul(
            temp1,
            valueGated,
            valueGated,
            count);

        SyncVectorToScalar();

        const float valueSquareSum =
            SumLocal(temp1);

        const float valueMeanSquare =
            valueSquareSum
            / static_cast<float>(
                hiddenSize_);

        const float valueRms =
            ScalarSqrt(
                valueMeanSquare
                + normEps_);

        const float invValueRms =
            1.0f / valueRms;

        AscendC::Muls(
            normedValueGated,
            valueGated,
            invValueRms,
            count);

        AscendC::Mul(
            normedValueGated,
            normedValueGated,
            convNormLocal,
            count);

        // -------------------------------------------------
        // FP32 workspaces -> GM
        // -------------------------------------------------

        auto valueGatedOut =
            valueGatedOutQueue_
                .AllocTensor<float>();

        auto normedValueGatedOut =
            normedValueGatedOutQueue_
                .AllocTensor<float>();

        AscendC::Adds(
            valueGatedOut,
            valueGated,
            0.0f,
            count);

        AscendC::Adds(
            normedValueGatedOut,
            normedValueGated,
            0.0f,
            count);

        valueGatedOutQueue_
            .EnQue(valueGatedOut);

        normedValueGatedOutQueue_
            .EnQue(normedValueGatedOut);

        valueGatedOut =
            valueGatedOutQueue_
                .DeQue<float>();

        normedValueGatedOut =
            normedValueGatedOutQueue_
                .DeQue<float>();

        AscendC::DataCopyPad(
            valueGatedGm_[
                activationOffset],
            valueGatedOut,
            floatCopyParams);

        AscendC::DataCopyPad(
            normedValueGatedGm_[
                activationOffset],
            normedValueGatedOut,
            floatCopyParams);

        valueGatedOutQueue_
            .FreeTensor(
                valueGatedOut);

        normedValueGatedOutQueue_
            .FreeTensor(
                normedValueGatedOut);

        keysQueue_
            .FreeTensor(keysLocal);

        hiddenQueue_
            .FreeTensor(hiddenLocal);

        valueQueue_
            .FreeTensor(valueLocal);

        norm1Queue_
            .FreeTensor(norm1Local);

        norm2Queue_
            .FreeTensor(norm2Local);

        convNormQueue_
            .FreeTensor(convNormLocal);
    }

private:
    AscendC::TPipe pipe_;

    AscendC::GlobalTensor<bfloat16_t>
        keysGm_;

    AscendC::GlobalTensor<bfloat16_t>
        hiddenGm_;

    AscendC::GlobalTensor<bfloat16_t>
        valueGm_;

    AscendC::GlobalTensor<float>
        norm1WeightGm_;

    AscendC::GlobalTensor<float>
        norm2WeightGm_;

    AscendC::GlobalTensor<float>
        convNormWeightGm_;

    AscendC::GlobalTensor<float>
        valueGatedGm_;

    AscendC::GlobalTensor<float>
        normedValueGatedGm_;

    AscendC::TQue<
        AscendC::QuePosition::VECIN,
        1> keysQueue_;

    AscendC::TQue<
        AscendC::QuePosition::VECIN,
        1> hiddenQueue_;

    AscendC::TQue<
        AscendC::QuePosition::VECIN,
        1> valueQueue_;

    AscendC::TQue<
        AscendC::QuePosition::VECIN,
        1> norm1Queue_;

    AscendC::TQue<
        AscendC::QuePosition::VECIN,
        1> norm2Queue_;

    AscendC::TQue<
        AscendC::QuePosition::VECIN,
        1> convNormQueue_;

    AscendC::TQue<
        AscendC::QuePosition::VECOUT,
        1> valueGatedOutQueue_;

    AscendC::TQue<
        AscendC::QuePosition::VECOUT,
        1> normedValueGatedOutQueue_;

    AscendC::TBuf<
        AscendC::TPosition::VECCALC>
        keysFloatBuf_;

    AscendC::TBuf<
        AscendC::TPosition::VECCALC>
        hiddenFloatBuf_;

    AscendC::TBuf<
        AscendC::TPosition::VECCALC>
        valueFloatBuf_;

    AscendC::TBuf<
        AscendC::TPosition::VECCALC>
        temp1Buf_;

    AscendC::TBuf<
        AscendC::TPosition::VECCALC>
        temp2Buf_;

    AscendC::TBuf<
        AscendC::TPosition::VECCALC>
        valueGatedBuf_;

    AscendC::TBuf<
        AscendC::TPosition::VECCALC>
        normedValueGatedBuf_;

    AscendC::TBuf<
        AscendC::TPosition::VECCALC>
        scalarInBuf_;

    AscendC::TBuf<
        AscendC::TPosition::VECCALC>
        scalarOutBuf_;

    int64_t batchSize_;
    int64_t seqLength_;
    int64_t hcMult_;
    int64_t hiddenSize_;
    int64_t totalItems_;
    int64_t itemsPerBlock_;

    float normEps_;
};


__global__ __aicore__ void engram_stage_a_kernel(
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
    int64_t itemsPerBlock)
{
    EngramStageAKernel op;

    op.Init(
        keys,
        hiddenStates,
        value,
        norm1Weight,
        norm2Weight,
        convNormWeight,
        valueGated,
        normedValueGated,
        batchSize,
        seqLength,
        hcMult,
        hiddenSize,
        normEps,
        itemsPerBlock);

    op.Process();
}


std::tuple<int64_t, int64_t>
calc_engram_stage_a_tiling_params(
    int64_t totalItems)
{
    auto platform =
        platform_ascendc::
            PlatformAscendCManager::
                GetInstance();

    int64_t coreNum =
        platform->GetCoreNumAiv();

    if (coreNum <= 0) {
        coreNum = 1;
    }

    int64_t numBlocks =
        totalItems;

    if (numBlocks > coreNum) {
        numBlocks = coreNum;
    }

    if (numBlocks <= 0) {
        numBlocks = 1;
    }

    const int64_t itemsPerBlock =
        (totalItems + numBlocks - 1)
        / numBlocks;

    return std::make_tuple(
        numBlocks,
        itemsPerBlock);
}


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
    void* stream)
{
    engram_stage_a_kernel
        <<<numBlocks, nullptr, stream>>>(
            keys,
            hiddenStates,
            value,
            norm1Weight,
            norm2Weight,
            convNormWeight,
            valueGated,
            normedValueGated,
            batchSize,
            seqLength,
            hcMult,
            hiddenSize,
            normEps,
            itemsPerBlock);
}

}
