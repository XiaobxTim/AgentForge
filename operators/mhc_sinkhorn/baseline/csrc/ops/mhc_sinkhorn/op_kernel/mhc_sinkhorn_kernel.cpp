#include <algorithm>
#include <tuple>

#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

constexpr uint32_t MAX_HC_MULT = 16;
constexpr uint32_t MAX_MATRIX_ELEMENTS =
    MAX_HC_MULT * MAX_HC_MULT;

template <typename T>
class MhcSinkhornKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR input,
        GM_ADDR output,
        int64_t batchSize,
        int64_t hcMult,
        int64_t iterStep,
        float eps,
        int64_t matricesPerBlock)
    {
        batchSize_ = batchSize;
        hcMult_ = hcMult;
        iterStep_ = iterStep;
        eps_ = eps;
        matricesPerBlock_ = matricesPerBlock;

        matrixElements_ =
            hcMult_ * hcMult_;

        inputGm_.SetGlobalBuffer(
            (__gm__ T*)input);

        outputGm_.SetGlobalBuffer(
            (__gm__ T*)output);

        pipe_.InitBuffer(
            inputQueue_,
            1,
            MAX_MATRIX_ELEMENTS * sizeof(T));

        pipe_.InitBuffer(
            outputQueue_,
            1,
            MAX_MATRIX_ELEMENTS * sizeof(T));

        pipe_.InitBuffer(
            matrixBuf_,
            MAX_MATRIX_ELEMENTS * sizeof(T));

        // Two aligned row buffers.
        // hc_mult <= 16, so 16 fp32 values = 64 bytes.
        pipe_.InitBuffer(
            shiftedRowBuf_,
            MAX_HC_MULT * sizeof(T));

        pipe_.InitBuffer(
            expRowBuf_,
            MAX_HC_MULT * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        const int64_t blockId =
            static_cast<int64_t>(
                AscendC::GetBlockIdx());

        const int64_t firstMatrix =
            blockId * matricesPerBlock_;

        if (firstMatrix >= batchSize_) {
            return;
        }

        int64_t lastMatrix =
            firstMatrix + matricesPerBlock_;

        if (lastMatrix > batchSize_) {
            lastMatrix = batchSize_;
        }

        for (int64_t matrixId = firstMatrix;
             matrixId < lastMatrix;
             ++matrixId) {

            ProcessOneMatrix(matrixId);
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

    __aicore__ inline void ProcessOneMatrix(
        int64_t matrixId)
    {
        const int64_t gmOffset =
            matrixId * matrixElements_;

        const uint32_t matrixBytes =
            static_cast<uint32_t>(
                matrixElements_ * sizeof(T));

        AscendC::DataCopyExtParams copyParams{
            1,
            matrixBytes,
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

        // --------------------------------------------------
        // GM -> UB
        // --------------------------------------------------

        auto inputLocal =
            inputQueue_.AllocTensor<T>();

        AscendC::DataCopyPad(
            inputLocal,
            inputGm_[gmOffset],
            copyParams,
            padParams);

        inputQueue_.EnQue(inputLocal);

        inputLocal =
            inputQueue_.DeQue<T>();

        auto matrixLocal =
            matrixBuf_.Get<T>();

        // Vector copy is used here so that the queue synchronizes
        // the GM load with the vector pipeline.
        AscendC::Adds(
            matrixLocal,
            inputLocal,
            static_cast<T>(0.0f),
            static_cast<int32_t>(
                matrixElements_));

        SyncVectorToScalar();

        inputQueue_.FreeTensor(
            inputLocal);

        // --------------------------------------------------
        // Iteration 1:
        // stable row softmax + eps
        // --------------------------------------------------

        FirstRowSoftmax();

        // First column normalization.
        NormalizeColumns();

        // --------------------------------------------------
        // Iterations 2 .. iter_step
        // --------------------------------------------------

        for (int64_t iter = 1;
             iter < iterStep_;
             ++iter) {

            NormalizeRows();
            NormalizeColumns();
        }

        // --------------------------------------------------
        // UB -> GM
        // --------------------------------------------------

        auto outputLocal =
            outputQueue_.AllocTensor<T>();

        // matrixLocal has been updated by scalar instructions.
        // Synchronize scalar -> vector before copying it into
        // the VECOUT queue.
        SyncScalarToVector();

        AscendC::Adds(
            outputLocal,
            matrixLocal,
            static_cast<T>(0.0f),
            static_cast<int32_t>(
                matrixElements_));

        outputQueue_.EnQue(
            outputLocal);

        outputLocal =
            outputQueue_.DeQue<T>();

        AscendC::DataCopyPad(
            outputGm_[gmOffset],
            outputLocal,
            copyParams);

        outputQueue_.FreeTensor(
            outputLocal);
    }

    __aicore__ inline void FirstRowSoftmax()
    {
        auto matrixLocal =
            matrixBuf_.Get<T>();

        auto shiftedRow =
            shiftedRowBuf_.Get<T>();

        auto expRow =
            expRowBuf_.Get<T>();

        for (int64_t row = 0;
             row < hcMult_;
             ++row) {

            const int64_t rowOffset =
                row * hcMult_;

            // ----------------------------------------------
            // Stable row max.
            // NaN is propagated deliberately.
            // ----------------------------------------------

            T rowMax =
                matrixLocal.GetValue(
                    static_cast<uint32_t>(
                        rowOffset));

            if (!(rowMax != rowMax)) {
                for (int64_t col = 1;
                     col < hcMult_;
                     ++col) {

                    T value =
                        matrixLocal.GetValue(
                            static_cast<uint32_t>(
                                rowOffset + col));

                    if (value != value) {
                        rowMax = value;
                        break;
                    }

                    if (value > rowMax) {
                        rowMax = value;
                    }
                }
            }

            // ----------------------------------------------
            // comb - row_max
            //
            // Fill an aligned temporary row because arbitrary
            // matrix rows are not necessarily 32-byte aligned
            // when hc_mult is 3, 6, 12, etc.
            // ----------------------------------------------

            for (int64_t col = 0;
                 col < hcMult_;
                 ++col) {

                T value =
                    matrixLocal.GetValue(
                        static_cast<uint32_t>(
                            rowOffset + col));

                shiftedRow.SetValue(
                    static_cast<uint32_t>(col),
                    value - rowMax);
            }

            // scalar -> vector
            SyncScalarToVector();

            AscendC::Exp(
                expRow,
                shiftedRow,
                static_cast<int32_t>(
                    hcMult_));

            // vector -> scalar
            SyncVectorToScalar();

            // ----------------------------------------------
            // row_sum(exp(...))
            // ----------------------------------------------

            T rowSum =
                static_cast<T>(0.0f);

            for (int64_t col = 0;
                 col < hcMult_;
                 ++col) {

                rowSum +=
                    expRow.GetValue(
                        static_cast<uint32_t>(
                            col));
            }

            // Golden:
            //
            // comb = comb / row_sum + eps
            //
            for (int64_t col = 0;
                 col < hcMult_;
                 ++col) {

                const T value =
                    expRow.GetValue(
                        static_cast<uint32_t>(
                            col));

                matrixLocal.SetValue(
                    static_cast<uint32_t>(
                        rowOffset + col),
                    value / rowSum + eps_);
            }
        }
    }

    __aicore__ inline void NormalizeRows()
    {
        auto matrixLocal =
            matrixBuf_.Get<T>();

        for (int64_t row = 0;
             row < hcMult_;
             ++row) {

            const int64_t rowOffset =
                row * hcMult_;

            T rowSum =
                static_cast<T>(0.0f);

            for (int64_t col = 0;
                 col < hcMult_;
                 ++col) {

                rowSum +=
                    matrixLocal.GetValue(
                        static_cast<uint32_t>(
                            rowOffset + col));
            }

            const T denom =
                rowSum + eps_;

            for (int64_t col = 0;
                 col < hcMult_;
                 ++col) {

                const uint32_t index =
                    static_cast<uint32_t>(
                        rowOffset + col);

                const T value =
                    matrixLocal.GetValue(
                        index);

                matrixLocal.SetValue(
                    index,
                    value / denom);
            }
        }
    }

    __aicore__ inline void NormalizeColumns()
    {
        auto matrixLocal =
            matrixBuf_.Get<T>();

        for (int64_t col = 0;
             col < hcMult_;
             ++col) {

            T colSum =
                static_cast<T>(0.0f);

            for (int64_t row = 0;
                 row < hcMult_;
                 ++row) {

                const uint32_t index =
                    static_cast<uint32_t>(
                        row * hcMult_ + col);

                colSum +=
                    matrixLocal.GetValue(
                        index);
            }

            const T denom =
                colSum + eps_;

            for (int64_t row = 0;
                 row < hcMult_;
                 ++row) {

                const uint32_t index =
                    static_cast<uint32_t>(
                        row * hcMult_ + col);

                const T value =
                    matrixLocal.GetValue(
                        index);

                matrixLocal.SetValue(
                    index,
                    value / denom);
            }
        }
    }

private:
    AscendC::TPipe pipe_;

    AscendC::GlobalTensor<T>
        inputGm_;

    AscendC::GlobalTensor<T>
        outputGm_;

    AscendC::TQue<
        AscendC::QuePosition::VECIN,
        1> inputQueue_;

    AscendC::TQue<
        AscendC::QuePosition::VECOUT,
        1> outputQueue_;

    AscendC::TBuf<
        AscendC::TPosition::VECCALC>
        matrixBuf_;

    AscendC::TBuf<
        AscendC::TPosition::VECCALC>
        shiftedRowBuf_;

    AscendC::TBuf<
        AscendC::TPosition::VECCALC>
        expRowBuf_;

    int64_t batchSize_;
    int64_t hcMult_;
    int64_t iterStep_;
    int64_t matricesPerBlock_;
    int64_t matrixElements_;

    T eps_;
};


__global__ __aicore__ void mhc_sinkhorn_kernel(
    GM_ADDR input,
    GM_ADDR output,
    int64_t batchSize,
    int64_t hcMult,
    int64_t iterStep,
    float eps,
    int64_t matricesPerBlock)
{
    MhcSinkhornKernel<float> op;

    op.Init(
        input,
        output,
        batchSize,
        hcMult,
        iterStep,
        eps,
        matricesPerBlock);

    op.Process();
}


std::tuple<int64_t, int64_t>
calc_mhc_sinkhorn_tiling_params(
    int64_t batchSize)
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
        batchSize;

    if (numBlocks > coreNum) {
        numBlocks = coreNum;
    }

    if (numBlocks <= 0) {
        numBlocks = 1;
    }

    const int64_t matricesPerBlock =
        (batchSize + numBlocks - 1)
        / numBlocks;

    return std::make_tuple(
        numBlocks,
        matricesPerBlock);
}


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
    void* stream)
{
    mhc_sinkhorn_kernel
        <<<numBlocks, nullptr, stream>>>(
            input,
            output,
            batchSize,
            hcMult,
            iterStep,
            eps,
            matricesPerBlock);
}

}
