#include "kernel_operator.h"
#include "adv_api/activation/swish.h"
#include "engram_gate_fusion_launch.h"

class EngramStageBDecodeKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR valueGated,
        GM_ADDR normedValueGated,
        GM_ADDR convWeight,
        GM_ADDR convState,
        GM_ADDR output,
        GM_ADDR convStateOut,
        int64_t B,
        int64_t L,
        int64_t HC,
        int64_t D,
        int64_t K,
        int64_t dilation,
        int64_t itemsPerBlock)
    {
        B_ = B;
        L_ = L;
        HC_ = HC;
        D_ = D;
        K_ = K;
        dilation_ = dilation;
        itemsPerBlock_ = itemsPerBlock;

        stateLen_ =
            (K_ - 1) * dilation_;

        totalItems_ =
            B_ * L_ * HC_;

        valueGatedGm_.SetGlobalBuffer(
            (__gm__ float*)valueGated);

        normedVgGm_.SetGlobalBuffer(
            (__gm__ float*)normedValueGated);

        convWeightGm_.SetGlobalBuffer(
            (__gm__ float*)convWeight);

        convStateGm_.SetGlobalBuffer(
            (__gm__ bfloat16_t*)convState);

        outputGm_.SetGlobalBuffer(
            (__gm__ bfloat16_t*)output);

        stateOutGm_.SetGlobalBuffer(
            (__gm__ bfloat16_t*)convStateOut);

        const uint32_t floatVecBytes =
            static_cast<uint32_t>(
                D_ * sizeof(float));

        const uint32_t bf16VecBytes =
            static_cast<uint32_t>(
                D_ * sizeof(bfloat16_t));

        const uint32_t weightBytes =
            static_cast<uint32_t>(
                D_ * K_ * sizeof(float));

        const uint32_t stateBytes =
            static_cast<uint32_t>(
                D_ * stateLen_
                * sizeof(bfloat16_t));

        pipe_.InitBuffer(
            currentQueue_,
            1,
            floatVecBytes);

        pipe_.InitBuffer(
            residualQueue_,
            1,
            floatVecBytes);

        pipe_.InitBuffer(
            weightQueue_,
            1,
            weightBytes);

        pipe_.InitBuffer(
            outputQueue_,
            1,
            bf16VecBytes);

        pipe_.InitBuffer(
            sourceBuf_,
            floatVecBytes);

        pipe_.InitBuffer(
            weightBlockBuf_,
            weightBytes);

        pipe_.InitBuffer(
            weightVectorBuf_,
            floatVecBytes);

        pipe_.InitBuffer(
            accumBuf_,
            floatVecBytes);

        pipe_.InitBuffer(
            temp1Buf_,
            floatVecBytes);

        pipe_.InitBuffer(
            temp2Buf_,
            floatVecBytes);

        pipe_.InitBuffer(
            stateBuildBuf_,
            stateBytes);
    }

    __aicore__ inline void Process()
    {
        const int64_t blockId =
            static_cast<int64_t>(
                AscendC::GetBlockIdx());

        const int64_t first =
            blockId * itemsPerBlock_;

        if (first >= totalItems_) {
            return;
        }

        int64_t last =
            first + itemsPerBlock_;

        if (last > totalItems_) {
            last = totalItems_;
        }

        for (int64_t item = first;
             item < last;
             ++item) {
            ProcessOneItem(item);
        }
    }

private:
    __aicore__ inline void
    SyncVectorToScalar()
    {
        AscendC::SetFlag<
            AscendC::HardEvent::V_S>(1);

        AscendC::WaitFlag<
            AscendC::HardEvent::V_S>(1);
    }

    __aicore__ inline void
    SyncScalarToVector()
    {
        AscendC::SetFlag<
            AscendC::HardEvent::S_V>(0);

        AscendC::WaitFlag<
            AscendC::HardEvent::S_V>(0);
    }

    __aicore__ inline int64_t
    ActivationOffset(
        int64_t b,
        int64_t l,
        int64_t hc) const
    {
        return (
            (
                b * L_ + l
            ) * HC_ + hc
        ) * D_;
    }

    __aicore__ inline int64_t
    StateOffset(
        int64_t b,
        int64_t hc,
        int64_t d,
        int64_t s) const
    {
        return (
            (
                (
                    b * HC_ + hc
                ) * D_ + d
            ) * stateLen_
            + s
        );
    }

    __aicore__ inline void
    LoadCurrentSource(
        int64_t b,
        int64_t l,
        int64_t hc)
    {
        const int64_t offset =
            ActivationOffset(
                b, l, hc);

        const uint32_t bytes =
            static_cast<uint32_t>(
                D_ * sizeof(float));

        AscendC::DataCopyExtParams params{
            1,
            bytes,
            0,
            0,
            0
        };

        AscendC::DataCopyPadExtParams<
            float> pad{
                false, 0, 0, 0
            };

        auto local =
            currentQueue_
                .AllocTensor<float>();

        AscendC::DataCopyPad(
            local,
            normedVgGm_[offset],
            params,
            pad);

        currentQueue_.EnQue(local);

        local =
            currentQueue_
                .DeQue<float>();

        auto source =
            sourceBuf_.Get<float>();

        AscendC::Adds(
            source,
            local,
            0.0f,
            static_cast<int32_t>(D_));

        currentQueue_.FreeTensor(local);
    }

    __aicore__ inline void
    LoadStateSource(
        int64_t b,
        int64_t hc,
        int64_t stateIndex)
    {
        auto source =
            sourceBuf_.Get<float>();

        for (int64_t d = 0;
             d < D_;
             ++d) {

            const bfloat16_t v =
                convStateGm_.GetValue(
                    static_cast<uint64_t>(
                        StateOffset(
                            b,
                            hc,
                            d,
                            stateIndex)));

            source.SetValue(
                static_cast<uint32_t>(d),
                AscendC::Cast(v));
        }

        SyncScalarToVector();
    }

    __aicore__ inline void
    LoadWeightBlock(int64_t hc)
    {
        const int64_t offset =
            hc * D_ * K_;

        const uint32_t bytes =
            static_cast<uint32_t>(
                D_ * K_
                * sizeof(float));

        AscendC::DataCopyExtParams params{
            1,
            bytes,
            0,
            0,
            0
        };

        AscendC::DataCopyPadExtParams<
            float> pad{
                false, 0, 0, 0
            };

        auto local =
            weightQueue_
                .AllocTensor<float>();

        AscendC::DataCopyPad(
            local,
            convWeightGm_[offset],
            params,
            pad);

        weightQueue_.EnQue(local);

        local =
            weightQueue_
                .DeQue<float>();

        auto block =
            weightBlockBuf_.Get<float>();

        AscendC::Adds(
            block,
            local,
            0.0f,
            static_cast<int32_t>(
                D_ * K_));

        SyncVectorToScalar();

        weightQueue_.FreeTensor(local);
    }

    __aicore__ inline void
    BuildWeightVector(int64_t k)
    {
        auto block =
            weightBlockBuf_.Get<float>();

        auto vector =
            weightVectorBuf_.Get<float>();

        for (int64_t d = 0;
             d < D_;
             ++d) {

            vector.SetValue(
                static_cast<uint32_t>(d),
                block.GetValue(
                    static_cast<uint32_t>(
                        d * K_ + k)));
        }

        SyncScalarToVector();
    }

    __aicore__ inline void
    WriteStateOut(
        int64_t b,
        int64_t hc)
    {
        auto stateBuild =
            stateBuildBuf_
                .Get<bfloat16_t>();

        auto source =
            sourceBuf_.Get<float>();

        // Golden:
        //
        // x_cat = cat([old_state, current_x], dim=-1)
        // state_out = x_cat[..., -stateLen:]
        //
        // Last stateLen positions begin at x_cat index L.
        for (int64_t s = 0;
             s < stateLen_;
             ++s) {

            const int64_t catIndex =
                L_ + s;

            if (catIndex < stateLen_) {
                // Still comes from the old state.
                const int64_t oldStateIndex =
                    catIndex;

                for (int64_t d = 0;
                     d < D_;
                     ++d) {

                    const bfloat16_t v =
                        convStateGm_.GetValue(
                            static_cast<uint64_t>(
                                StateOffset(
                                    b,
                                    hc,
                                    d,
                                    oldStateIndex)));

                    stateBuild.SetValue(
                        static_cast<uint32_t>(
                            d * stateLen_ + s),
                        v);
                }
            } else {
                // Comes from current normed_vg sequence.
                const int64_t currentToken =
                    catIndex -
                    stateLen_;

                LoadCurrentSource(
                    b,
                    currentToken,
                    hc);

                SyncVectorToScalar();

                for (int64_t d = 0;
                     d < D_;
                     ++d) {

                    stateBuild.SetValue(
                        static_cast<uint32_t>(
                            d * stateLen_ + s),
                        AscendC::Cast(
                            source.GetValue(
                                static_cast<uint32_t>(
                                    d))));
                }
            }
        }

        const int64_t gmOffset =
            (
                b * HC_ + hc
            ) *
            D_ *
            stateLen_;

        const uint32_t bytes =
            static_cast<uint32_t>(
                D_ *
                stateLen_ *
                sizeof(bfloat16_t));

        AscendC::DataCopyExtParams params{
            1,
            bytes,
            0,
            0,
            0
        };

        AscendC::DataCopyPad(
            stateOutGm_[gmOffset],
            stateBuild,
            params);
    }

    __aicore__ inline void
    ProcessOneItem(int64_t item)
    {
        const int64_t hc =
            item % HC_;

        const int64_t token =
            item / HC_;

        const int64_t l =
            token % L_;

        const int64_t b =
            token / L_;

        const int32_t count =
            static_cast<int32_t>(D_);

        const int64_t outputOffset =
            ActivationOffset(
                b, l, hc);

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

        // Causal dilated depthwise cross-correlation.
        for (int64_t k = 0;
             k < K_;
             ++k) {

            const int64_t historyDistance =
                (
                    K_ - 1 - k
                ) * dilation_;

            const int64_t sourceToken =
                l - historyDistance;

            if (sourceToken >= 0) {
                LoadCurrentSource(
                    b,
                    sourceToken,
                    hc);
            } else {
                const int64_t stateIndex =
                    stateLen_
                    + sourceToken;

                LoadStateSource(
                    b,
                    hc,
                    stateIndex);
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

        // SiLU(x) = Swish(x, beta=1)
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


        // Residual value_gated.
        const uint32_t floatBytes =
            static_cast<uint32_t>(
                D_ * sizeof(float));

        AscendC::DataCopyExtParams params{
            1,
            floatBytes,
            0,
            0,
            0
        };

        AscendC::DataCopyPadExtParams<
            float> pad{
                false, 0, 0, 0
            };

        auto residual =
            residualQueue_
                .AllocTensor<float>();

        AscendC::DataCopyPad(
            residual,
            valueGatedGm_[outputOffset],
            params,
            pad);

        residualQueue_.EnQue(
            residual);

        residual =
            residualQueue_
                .DeQue<float>();

        AscendC::Add(
            accum,
            accum,
            residual,
            count);

        residualQueue_
            .FreeTensor(residual);

        // FP32 -> BF16 output.
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

        AscendC::DataCopyExtParams outParams{
            1,
            static_cast<uint32_t>(
                D_ * sizeof(bfloat16_t)),
            0,
            0,
            0
        };

        AscendC::DataCopyPad(
            outputGm_[outputOffset],
            outputLocal,
            outParams);

        outputQueue_
            .FreeTensor(outputLocal);

        // One item per [b,hc] writes updated state.
        if (l == L_ - 1) {
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
        normedVgGm_;

    AscendC::GlobalTensor<float>
        convWeightGm_;

    AscendC::GlobalTensor<bfloat16_t>
        convStateGm_;

    AscendC::GlobalTensor<bfloat16_t>
        outputGm_;

    AscendC::GlobalTensor<bfloat16_t>
        stateOutGm_;

    AscendC::TQue<
        AscendC::QuePosition::VECIN,
        1> currentQueue_;

    AscendC::TQue<
        AscendC::QuePosition::VECIN,
        1> residualQueue_;

    AscendC::TQue<
        AscendC::QuePosition::VECIN,
        1> weightQueue_;

    AscendC::TQue<
        AscendC::QuePosition::VECOUT,
        1> outputQueue_;

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

    int64_t B_;
    int64_t L_;
    int64_t HC_;
    int64_t D_;
    int64_t K_;
    int64_t dilation_;
    int64_t stateLen_;
    int64_t totalItems_;
    int64_t itemsPerBlock_;
};


__global__ __aicore__ void
engram_stage_b_decode_kernel(
    GM_ADDR valueGated,
    GM_ADDR normedValueGated,
    GM_ADDR convWeight,
    GM_ADDR convState,
    GM_ADDR output,
    GM_ADDR convStateOut,
    int64_t B,
    int64_t L,
    int64_t HC,
    int64_t D,
    int64_t K,
    int64_t dilation,
    int64_t itemsPerBlock)
{
    EngramStageBDecodeKernel op;

    op.Init(
        valueGated,
        normedValueGated,
        convWeight,
        convState,
        output,
        convStateOut,
        B,
        L,
        HC,
        D,
        K,
        dilation,
        itemsPerBlock);

    op.Process();
}


extern "C" {

void launch_engram_stage_b_decode(
    GM_ADDR valueGated,
    GM_ADDR normedValueGated,
    GM_ADDR convWeight,
    GM_ADDR convState,
    GM_ADDR output,
    GM_ADDR convStateOut,
    int64_t B,
    int64_t L,
    int64_t HC,
    int64_t D,
    int64_t K,
    int64_t dilation,
    int64_t numBlocks,
    int64_t itemsPerBlock,
    void* stream)
{
    engram_stage_b_decode_kernel
        <<<numBlocks, nullptr, stream>>>(
            valueGated,
            normedValueGated,
            convWeight,
            convState,
            output,
            convStateOut,
            B,
            L,
            HC,
            D,
            K,
            dilation,
            itemsPerBlock);
}

}
