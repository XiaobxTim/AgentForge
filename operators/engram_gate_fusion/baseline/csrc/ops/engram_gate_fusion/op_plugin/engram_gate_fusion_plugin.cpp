#include <tuple>

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/engram_gate_fusion_launch.h"
namespace cann_bench {

// ============================================================================
// Final AgentForge EngramGateFusion interface
// ============================================================================

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def(
        "engram_gate_fusion("
        "Tensor keys, "
        "Tensor hidden_states, "
        "Tensor value, "
        "Tensor norm1_weight, "
        "Tensor norm2_weight, "
        "Tensor conv_norm_weight, "
        "Tensor conv_weight, "
        "Tensor? conv_state=None, "
        "int hc_mult=4, "
        "int hidden_size=1024, "
        "int kernel_size=4, "
        "int dilation=3, "
        "float norm_eps=1e-5"
        ") -> (Tensor, Tensor)");
}


static void check_engram_inputs(
    const torch::Tensor& keys,
    const torch::Tensor& hiddenStates,
    const torch::Tensor& value,
    const torch::Tensor& norm1Weight,
    const torch::Tensor& norm2Weight,
    const torch::Tensor& convNormWeight,
    const torch::Tensor& convWeight,
    const c10::optional<torch::Tensor>& convState,
    int64_t hcMult,
    int64_t hiddenSize,
    int64_t kernelSize,
    int64_t dilation)
{
    TORCH_CHECK(
        keys.dim() == 4,
        "keys must have shape [B,L,HC,D].");

    TORCH_CHECK(
        hiddenStates.sizes() == keys.sizes(),
        "hidden_states shape must match keys.");

    TORCH_CHECK(
        keys.scalar_type() == torch::kBFloat16 &&
        hiddenStates.scalar_type() == torch::kBFloat16,
        "keys and hidden_states must be bfloat16.");

    const int64_t B = keys.size(0);
    const int64_t L = keys.size(1);

    TORCH_CHECK(
        keys.size(2) == hcMult,
        "HC must equal hc_mult.");

    TORCH_CHECK(
        keys.size(3) == hiddenSize,
        "D must equal hidden_size.");

    TORCH_CHECK(
        value.dim() == 3 &&
        value.size(0) == B &&
        value.size(1) == L &&
        value.size(2) == hiddenSize,
        "value shape mismatch.");

    TORCH_CHECK(
        value.scalar_type() == torch::kBFloat16,
        "value must be bfloat16.");

    TORCH_CHECK(
        norm1Weight.dim() == 2 &&
        norm1Weight.size(0) == hcMult &&
        norm1Weight.size(1) == hiddenSize,
        "norm1_weight shape mismatch.");

    TORCH_CHECK(
        norm2Weight.sizes() == norm1Weight.sizes() &&
        convNormWeight.sizes() == norm1Weight.sizes(),
        "RMSNorm weight shapes must match [HC,D].");

    TORCH_CHECK(
        norm1Weight.scalar_type() == torch::kFloat32 &&
        norm2Weight.scalar_type() == torch::kFloat32 &&
        convNormWeight.scalar_type() == torch::kFloat32,
        "RMSNorm weights must be float32.");

    TORCH_CHECK(
        convWeight.dim() == 3 &&
        convWeight.size(0) == hcMult * hiddenSize &&
        convWeight.size(1) == 1 &&
        convWeight.size(2) == kernelSize,
        "conv_weight shape must be [HC*D,1,K].");

    TORCH_CHECK(
        convWeight.scalar_type() == torch::kFloat32,
        "conv_weight must be float32.");

    TORCH_CHECK(
        kernelSize >= 1,
        "kernel_size must be >= 1.");

    TORCH_CHECK(
        dilation >= 1,
        "dilation must be >= 1.");

    const int64_t stateLen =
        (kernelSize - 1) * dilation;

    if (convState.has_value()) {
        const auto& state =
            convState.value();

        TORCH_CHECK(
            state.dim() == 3 &&
            state.size(0) == B &&
            state.size(1) == hcMult * hiddenSize &&
            state.size(2) == stateLen,
            "conv_state shape mismatch.");

        TORCH_CHECK(
            state.scalar_type() == torch::kBFloat16,
            "conv_state must be bfloat16.");
    }

    if (L == 1) {
        TORCH_CHECK(
            convState.has_value(),
            "L=1 Decode requires conv_state.");
    }
}


std::tuple<
    torch::Tensor,
    torch::Tensor>
engram_gate_fusion_meta(
    const torch::Tensor& keys,
    const torch::Tensor& hiddenStates,
    const torch::Tensor& value,
    const torch::Tensor& norm1Weight,
    const torch::Tensor& norm2Weight,
    const torch::Tensor& convNormWeight,
    const torch::Tensor& convWeight,
    const c10::optional<torch::Tensor>& convState,
    int64_t hcMult,
    int64_t hiddenSize,
    int64_t kernelSize,
    int64_t dilation,
    double normEps)
{
    check_engram_inputs(
        keys,
        hiddenStates,
        value,
        norm1Weight,
        norm2Weight,
        convNormWeight,
        convWeight,
        convState,
        hcMult,
        hiddenSize,
        kernelSize,
        dilation);

    const int64_t B =
        keys.size(0);

    const int64_t L =
        keys.size(1);

    const int64_t stateLen =
        (kernelSize - 1) * dilation;

    auto output =
        torch::empty_like(keys);

    auto stateOptions =
        keys.options()
            .dtype(torch::kBFloat16);

    auto stateOut =
        torch::empty(
            {
                B,
                hcMult * hiddenSize,
                stateLen
            },
            stateOptions);

    return {
        output,
        stateOut
    };
}


TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl(
        "engram_gate_fusion",
        engram_gate_fusion_meta);
}


std::tuple<
    torch::Tensor,
    torch::Tensor>
engram_gate_fusion_npu(
    const torch::Tensor& keys,
    const torch::Tensor& hiddenStates,
    const torch::Tensor& value,
    const torch::Tensor& norm1Weight,
    const torch::Tensor& norm2Weight,
    const torch::Tensor& convNormWeight,
    const torch::Tensor& convWeight,
    const c10::optional<torch::Tensor>& convState,
    int64_t hcMult,
    int64_t hiddenSize,
    int64_t kernelSize,
    int64_t dilation,
    double normEps)
{
    const c10::OptionalDeviceGuard guard(
        keys.device());

    TORCH_CHECK(
        keys.is_contiguous() &&
        hiddenStates.is_contiguous() &&
        value.is_contiguous() &&
        norm1Weight.is_contiguous() &&
        norm2Weight.is_contiguous() &&
        convNormWeight.is_contiguous() &&
        convWeight.is_contiguous(),
        "EngramGateFusion baseline requires contiguous inputs.");

    if (convState.has_value()) {
        TORCH_CHECK(
            convState.value().is_contiguous(),
            "conv_state must be contiguous.");
    }

    check_engram_inputs(
        keys,
        hiddenStates,
        value,
        norm1Weight,
        norm2Weight,
        convNormWeight,
        convWeight,
        convState,
        hcMult,
        hiddenSize,
        kernelSize,
        dilation);

    const int64_t B =
        keys.size(0);

    const int64_t L =
        keys.size(1);

    const int64_t totalItems =
        B * L * hcMult;

    const int64_t stateLen =
        (kernelSize - 1) * dilation;

    auto fp32Options =
        keys.options()
            .dtype(torch::kFloat32);

    // Starter Baseline deliberately materializes two FP32 workspaces.
    auto valueGated =
        torch::empty(
            keys.sizes(),
            fp32Options);

    auto normedValueGated =
        torch::empty(
            keys.sizes(),
            fp32Options);

    auto output =
        torch::empty_like(keys);

    auto stateOut =
        torch::empty(
            {
                B,
                hcMult * hiddenSize,
                stateLen
            },
            keys.options()
                .dtype(torch::kBFloat16));

    int64_t stageABlocks;
    int64_t stageAItemsPerBlock;

    std::tie(
        stageABlocks,
        stageAItemsPerBlock) =
        calc_engram_stage_a_tiling_params(
            totalItems);

    int64_t stageBBlocks;
    int64_t stageBItemsPerBlock;

    std::tie(
        stageBBlocks,
        stageBItemsPerBlock) =
        calc_engram_stage_b_tiling_params(
            totalItems);

    auto stream =
        c10_npu::getCurrentNPUStream()
            .stream(false);

    const float normEpsFloat =
        static_cast<float>(normEps);

    auto keysPtr =
        (GM_ADDR)keys.data_ptr();

    auto hiddenPtr =
        (GM_ADDR)hiddenStates.data_ptr();

    auto valuePtr =
        (GM_ADDR)value.data_ptr();

    auto norm1Ptr =
        (GM_ADDR)norm1Weight.data_ptr();

    auto norm2Ptr =
        (GM_ADDR)norm2Weight.data_ptr();

    auto convNormPtr =
        (GM_ADDR)convNormWeight.data_ptr();

    auto convWeightPtr =
        (GM_ADDR)convWeight.data_ptr();

    auto valueGatedPtr =
        (GM_ADDR)valueGated.data_ptr();

    auto normedValueGatedPtr =
        (GM_ADDR)normedValueGated.data_ptr();

    auto outputPtr =
        (GM_ADDR)output.data_ptr();

    auto stateOutPtr =
        (GM_ADDR)stateOut.data_ptr();

    auto aclCall = [=]() -> int {
        // ----------------------------------------------------
        // Stage A
        // ----------------------------------------------------

        ENGRAM_STAGE_A_LAUNCH(
            keysPtr,
            hiddenPtr,
            valuePtr,
            norm1Ptr,
            norm2Ptr,
            convNormPtr,
            valueGatedPtr,
            normedValueGatedPtr,
            B,
            L,
            hcMult,
            hiddenSize,
            normEpsFloat,
            stageABlocks,
            stageAItemsPerBlock,
            stream);

        // ----------------------------------------------------
        // Stage B
        // ----------------------------------------------------

        if (convState.has_value()) {
            ENGRAM_STAGE_B_DECODE_LAUNCH(
                valueGatedPtr,
                normedValueGatedPtr,
                convWeightPtr,
                (GM_ADDR)convState.value().data_ptr(),
                outputPtr,
                stateOutPtr,
                B,
                L,
                hcMult,
                hiddenSize,
                kernelSize,
                dilation,
                stageBBlocks,
                stageBItemsPerBlock,
                stream);
        } else {
            ENGRAM_STAGE_B_PREFILL_LAUNCH(
                valueGatedPtr,
                normedValueGatedPtr,
                convWeightPtr,
                outputPtr,
                stateOutPtr,
                B,
                L,
                hcMult,
                hiddenSize,
                kernelSize,
                dilation,
                stageBBlocks,
                stageBItemsPerBlock,
                stream);
        }

        return 0;
    };

    at_npu::native::OpCommand::RunOpApi(
        "AgentForgeEngramGateFusion",
        aclCall);

    return {
        output,
        stateOut
    };
}


TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl(
        "engram_gate_fusion",
        engram_gate_fusion_npu);
}

} // namespace cann_bench
