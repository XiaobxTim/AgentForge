#include <algorithm>
#include <tuple>

#include "kernel_operator.h"
#include "adv_api/activation/swish.h"
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


// ============================================================================
// Engram Stage B — Prefill baseline
//
// Input:
//   valueGated        [B,L,HC,D] FP32
//   normedValueGated  [B,L,HC,D] FP32
//   convWeight        [HC*D,1,K] FP32
//
// Output:
//   output            [B,L,HC,D] BF16
//   convStateOut      [B,HC*D,stateLen] BF16
//
// This first Stage-B implementation handles Prefill only:
// historical positions before token 0 are treated as zero.
// ============================================================================

class EngramStageBPrefillKernel {
public:
    __aicore__ inline void Init(
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
        int64_t itemsPerBlock)
    {
        batchSize_ = batchSize;
        seqLength_ = seqLength;
        hcMult_ = hcMult;
        hiddenSize_ = hiddenSize;
        kernelSize_ = kernelSize;
        dilation_ = dilation;
        itemsPerBlock_ = itemsPerBlock;

        totalItems_ =
            batchSize_ *
            seqLength_ *
            hcMult_;

        stateLen_ =
            (kernelSize_ - 1) *
            dilation_;

        valueGatedGm_.SetGlobalBuffer(
            (__gm__ float*)valueGated);

        normedValueGatedGm_.SetGlobalBuffer(
            (__gm__ float*)normedValueGated);

        convWeightGm_.SetGlobalBuffer(
            (__gm__ float*)convWeight);

        outputGm_.SetGlobalBuffer(
            (__gm__ bfloat16_t*)output);

        convStateOutGm_.SetGlobalBuffer(
            (__gm__ bfloat16_t*)convStateOut);

        const uint32_t vectorFloatBytes =
            static_cast<uint32_t>(
                hiddenSize_ * sizeof(float));

        const uint32_t vectorBf16Bytes =
            static_cast<uint32_t>(
                hiddenSize_ *
                sizeof(bfloat16_t));

        const uint32_t weightBytes =
            static_cast<uint32_t>(
                hiddenSize_ *
                kernelSize_ *
                sizeof(float));

        const uint32_t stateBytes =
            static_cast<uint32_t>(
                hiddenSize_ *
                stateLen_ *
                sizeof(bfloat16_t));

        // Current [D] source vector from normedValueGated.
        pipe_.InitBuffer(
            sourceQueue_,
            1,
            vectorFloatBytes);

        // Current [D] residual vector.
        pipe_.InitBuffer(
            valueGatedQueue_,
            1,
            vectorFloatBytes);

        // Entire [D,K] conv-weight block for one HC.
        pipe_.InitBuffer(
            weightQueue_,
            1,
            weightBytes);

        // Final BF16 output vector [D].
        pipe_.InitBuffer(
            outputQueue_,
            1,
            vectorBf16Bytes);

        // Final state [D,stateLen].
        pipe_.InitBuffer(
            stateOutQueue_,
            1,
            stateBytes);

        pipe_.InitBuffer(
            sourceBuf_,
            vectorFloatBytes);

        pipe_.InitBuffer(
            weightBlockBuf_,
            weightBytes);

        pipe_.InitBuffer(
            weightVectorBuf_,
            vectorFloatBytes);

        pipe_.InitBuffer(
            accumBuf_,
            vectorFloatBytes);

        pipe_.InitBuffer(
            temp1Buf_,
            vectorFloatBytes);

        pipe_.InitBuffer(
            temp2Buf_,
            vectorFloatBytes);

        pipe_.InitBuffer(
            stateBuildBuf_,
            stateBytes);
    }

    __aicore__ inline void Process()
    {
        const int64_t blockId =
            static_cast<int64_t>(
                AscendC::GetBlockIdx());

        const int64_t firstItem =
            blockId *
            itemsPerBlock_;

        if (firstItem >= totalItems_) {
            return;
        }

        int64_t lastItem =
            firstItem +
            itemsPerBlock_;

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

    __aicore__ inline int64_t ActivationOffset(
        int64_t b,
        int64_t l,
        int64_t hc) const
    {
        return (
            (
                b * seqLength_ + l
            ) * hcMult_ + hc
        ) * hiddenSize_;
    }

    __aicore__ inline void BuildZeroSource()
    {
        auto source =
            sourceBuf_.Get<float>();

        for (int64_t d = 0;
             d < hiddenSize_;
             ++d) {
            source.SetValue(
                static_cast<uint32_t>(d),
                0.0f);
        }

        SyncScalarToVector();
    }

    __aicore__ inline void LoadCurrentSource(
        int64_t b,
        int64_t l,
        int64_t hc)
    {
        const int64_t offset =
            ActivationOffset(
                b,
                l,
                hc);

        const uint32_t bytes =
            static_cast<uint32_t>(
                hiddenSize_ *
                sizeof(float));

        AscendC::DataCopyExtParams params{
            1,
            bytes,
            0,
            0,
            0
        };

        AscendC::DataCopyPadExtParams<
            float> pad{
                false,
                0,
                0,
                0
            };

        auto sourceLocal =
            sourceQueue_
                .AllocTensor<float>();

        AscendC::DataCopyPad(
            sourceLocal,
            normedValueGatedGm_[offset],
            params,
            pad);

        sourceQueue_.EnQue(
            sourceLocal);

        sourceLocal =
            sourceQueue_
                .DeQue<float>();

        auto source =
            sourceBuf_.Get<float>();

        AscendC::Adds(
            source,
            sourceLocal,
            0.0f,
            static_cast<int32_t>(
                hiddenSize_));

        sourceQueue_.FreeTensor(
            sourceLocal);
    }

    __aicore__ inline void LoadWeightBlock(
        int64_t hc)
    {
        const int64_t offset =
            hc *
            hiddenSize_ *
            kernelSize_;

        const uint32_t bytes =
            static_cast<uint32_t>(
                hiddenSize_ *
                kernelSize_ *
                sizeof(float));

        AscendC::DataCopyExtParams params{
            1,
            bytes,
            0,
            0,
            0
        };

        AscendC::DataCopyPadExtParams<
            float> pad{
                false,
                0,
                0,
                0
            };

        auto weightLocal =
            weightQueue_
                .AllocTensor<float>();

        AscendC::DataCopyPad(
            weightLocal,
            convWeightGm_[offset],
            params,
            pad);

        weightQueue_.EnQue(
            weightLocal);

        weightLocal =
            weightQueue_
                .DeQue<float>();

        auto weightBlock =
            weightBlockBuf_.Get<float>();

        AscendC::Adds(
            weightBlock,
            weightLocal,
            0.0f,
            static_cast<int32_t>(
                hiddenSize_ *
                kernelSize_));

        SyncVectorToScalar();

        weightQueue_.FreeTensor(
            weightLocal);
    }

    __aicore__ inline void BuildWeightVector(
        int64_t k)
    {
        auto weightBlock =
            weightBlockBuf_.Get<float>();

        auto weightVector =
            weightVectorBuf_.Get<float>();

        for (int64_t d = 0;
             d < hiddenSize_;
             ++d) {

            const uint32_t srcIndex =
                static_cast<uint32_t>(
                    d * kernelSize_ + k);

            weightVector.SetValue(
                static_cast<uint32_t>(d),
                weightBlock.GetValue(
                    srcIndex));
        }

        SyncScalarToVector();
    }

    __aicore__ inline void WriteStateOut(
        int64_t b,
        int64_t hc)
    {
        if (stateLen_ <= 0) {
            return;
        }

        auto stateBuild =
            stateBuildBuf_.Get<bfloat16_t>();

        auto source =
            sourceBuf_.Get<float>();

        // Prefill:
        // x_cat = [zero padding, x]
        //
        // state_out is the last stateLen positions of x_cat.
        for (int64_t s = 0;
             s < stateLen_;
             ++s) {

            const int64_t t =
                seqLength_
                - stateLen_
                + s;

            if (t < 0) {
                const bfloat16_t zero =
                    AscendC::Cast(0.0f);

                for (int64_t d = 0;
                     d < hiddenSize_;
                     ++d) {

                    const uint32_t dstIndex =
                        static_cast<uint32_t>(
                            d * stateLen_ + s);

                    stateBuild.SetValue(
                        dstIndex,
                        zero);
                }

                continue;
            }

            LoadCurrentSource(
                b,
                t,
                hc);

            SyncVectorToScalar();

            for (int64_t d = 0;
                 d < hiddenSize_;
                 ++d) {

                const float value =
                    source.GetValue(
                        static_cast<uint32_t>(d));

                const uint32_t dstIndex =
                    static_cast<uint32_t>(
                        d * stateLen_ + s);

                stateBuild.SetValue(
                    dstIndex,
                    AscendC::Cast(value));
            }
        }

        const int64_t gmOffset =
            (
                b * hcMult_
                + hc
            ) *
            hiddenSize_ *
            stateLen_;

        const uint32_t bytes =
            static_cast<uint32_t>(
                hiddenSize_
                * stateLen_
                * sizeof(bfloat16_t));

        AscendC::DataCopyExtParams params{
            1,
            bytes,
            0,
            0,
            0
        };

        AscendC::DataCopyPad(
            convStateOutGm_[gmOffset],
            stateBuild,
            params);
    }

    __aicore__ inline void ProcessOneItem(
        int64_t item)
    {
        const int64_t hc =
            item % hcMult_;

        const int64_t token =
            item / hcMult_;

        const int64_t l =
            token % seqLength_;

        const int64_t b =
            token / seqLength_;

        const int64_t activationOffset =
            ActivationOffset(
                b,
                l,
                hc);

        const int32_t count =
            static_cast<int32_t>(
                hiddenSize_);

        // --------------------------------------------------
        // Load [D,K] conv weights for this HC.
        // --------------------------------------------------

        LoadWeightBlock(hc);

        auto source =
            sourceBuf_.Get<float>();

        auto weightVector =
            weightVectorBuf_.Get<float>();

        auto accum =
            accumBuf_.Get<float>();

        auto temp1 =
            temp1Buf_.Get<float>();

        auto temp2 =
            temp2Buf_.Get<float>();

        // --------------------------------------------------
        // PyTorch F.conv1d is cross-correlation:
        //
        // y[l] =
        // sum_k W[k] *
        // x_cat[l + k*dilation]
        //
        // After removing left padding:
        //
        // source_t =
        // l - (K - 1 - k) * dilation
        // --------------------------------------------------

        for (int64_t k = 0;
             k < kernelSize_;
             ++k) {

            const int64_t historyDistance =
                (
                    kernelSize_
                    - 1
                    - k
                ) *
                dilation_;

            const int64_t sourceToken =
                l -
                historyDistance;

            if (sourceToken < 0) {
                BuildZeroSource();
            } else {
                LoadCurrentSource(
                    b,
                    sourceToken,
                    hc);
            }

            BuildWeightVector(k);

            AscendC::Mul(
                temp1,
                source,
                weightVector,
                count);

            if (k == 0) {
                AscendC::Adds(
                    accum,
                    temp1,
                    0.0f,
                    count);
            } else {
                AscendC::Add(
                    accum,
                    accum,
                    temp1,
                    count);
            }
        }

        // --------------------------------------------------
        // SiLU(x) = Swish(x, beta=1)
        // --------------------------------------------------

        AscendC::Swish(
            temp1,
            accum,
            static_cast<uint32_t>(count),
            1.0f);

        AscendC::Adds(
            accum,
            temp1,
            0.0f,
            count);


        // --------------------------------------------------
        // Residual:
        // output = value_gated + conv_out
        // --------------------------------------------------

        const uint32_t floatBytes =
            static_cast<uint32_t>(
                hiddenSize_ *
                sizeof(float));

        AscendC::DataCopyExtParams
            floatCopyParams{
                1,
                floatBytes,
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

        auto valueGatedLocal =
            valueGatedQueue_
                .AllocTensor<float>();

        AscendC::DataCopyPad(
            valueGatedLocal,
            valueGatedGm_[
                activationOffset],
            floatCopyParams,
            floatPad);

        valueGatedQueue_.EnQue(
            valueGatedLocal);

        valueGatedLocal =
            valueGatedQueue_
                .DeQue<float>();

        AscendC::Add(
            accum,
            accum,
            valueGatedLocal,
            count);

        valueGatedQueue_.FreeTensor(
            valueGatedLocal);

        // --------------------------------------------------
        // FP32 -> BF16 final output
        // --------------------------------------------------

        auto outputLocal =
            outputQueue_
                .AllocTensor<bfloat16_t>();

        AscendC::Cast(
            outputLocal,
            accum,
            AscendC::RoundMode::CAST_RINT,
            count);

        outputQueue_.EnQue(
            outputLocal);

        outputLocal =
            outputQueue_
                .DeQue<bfloat16_t>();

        const uint32_t bf16Bytes =
            static_cast<uint32_t>(
                hiddenSize_ *
                sizeof(bfloat16_t));

        AscendC::DataCopyExtParams
            bf16CopyParams{
                1,
                bf16Bytes,
                0,
                0,
                0
            };

        AscendC::DataCopyPad(
            outputGm_[
                activationOffset],
            outputLocal,
            bf16CopyParams);

        outputQueue_.FreeTensor(
            outputLocal);

        // Exactly one item for each [b,hc]
        // generates conv_state_out.
        if (l == seqLength_ - 1) {
            WriteStateOut(
                b,
                hc);
        }
    }

private:
    AscendC::TPipe pipe_;

    AscendC::GlobalTensor<float>
        valueGatedGm_;

    AscendC::GlobalTensor<float>
        normedValueGatedGm_;

    AscendC::GlobalTensor<float>
        convWeightGm_;

    AscendC::GlobalTensor<bfloat16_t>
        outputGm_;

    AscendC::GlobalTensor<bfloat16_t>
        convStateOutGm_;

    AscendC::TQue<
        AscendC::QuePosition::VECIN,
        1> sourceQueue_;

    AscendC::TQue<
        AscendC::QuePosition::VECIN,
        1> valueGatedQueue_;

    AscendC::TQue<
        AscendC::QuePosition::VECIN,
        1> weightQueue_;

    AscendC::TQue<
        AscendC::QuePosition::VECOUT,
        1> outputQueue_;

    AscendC::TQue<
        AscendC::QuePosition::VECOUT,
        1> stateOutQueue_;

    AscendC::TBuf<
        AscendC::TPosition::VECCALC>
        sourceBuf_;

    AscendC::TBuf<
        AscendC::TPosition::VECCALC>
        weightBlockBuf_;

    AscendC::TBuf<
        AscendC::TPosition::VECCALC>
        weightVectorBuf_;

    AscendC::TBuf<
        AscendC::TPosition::VECCALC>
        accumBuf_;

    AscendC::TBuf<
        AscendC::TPosition::VECCALC>
        temp1Buf_;

    AscendC::TBuf<
        AscendC::TPosition::VECCALC>
        temp2Buf_;

    AscendC::TBuf<
        AscendC::TPosition::VECCALC>
        stateBuildBuf_;

    int64_t batchSize_;
    int64_t seqLength_;
    int64_t hcMult_;
    int64_t hiddenSize_;
    int64_t kernelSize_;
    int64_t dilation_;
    int64_t stateLen_;

    int64_t totalItems_;
    int64_t itemsPerBlock_;
};


__global__ __aicore__ void
engram_stage_b_prefill_kernel(
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
    int64_t itemsPerBlock)
{
    EngramStageBPrefillKernel op;

    op.Init(
        valueGated,
        normedValueGated,
        convWeight,
        output,
        convStateOut,
        batchSize,
        seqLength,
        hcMult,
        hiddenSize,
        kernelSize,
        dilation,
        itemsPerBlock);

    op.Process();
}


std::tuple<int64_t, int64_t>
calc_engram_stage_b_tiling_params(
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
        (
            totalItems +
            numBlocks -
            1
        ) /
        numBlocks;

    return std::make_tuple(
        numBlocks,
        itemsPerBlock);
}


extern "C" {

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
    void* stream)
{
    engram_stage_b_prefill_kernel
        <<<numBlocks, nullptr, stream>>>(
            valueGated,
            normedValueGated,
            convWeight,
            output,
            convStateOut,
            batchSize,
            seqLength,
            hcMult,
            hiddenSize,
            kernelSize,
            dilation,
            itemsPerBlock);
}

}
