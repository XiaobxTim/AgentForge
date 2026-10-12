#include <tuple>

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/engram_gate_fusion_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def(
        "engram_gate_stage_a("
        "Tensor keys, "
        "Tensor hidden_states, "
        "Tensor value, "
        "Tensor norm1_weight, "
        "Tensor norm2_weight, "
        "Tensor conv_norm_weight, "
        "int hc_mult=4, "
        "int hidden_size=1024, "
        "float norm_eps=1e-5"
        ") -> (Tensor, Tensor)");
}

static void check_stage_a_inputs(
    const torch::Tensor& keys,
    const torch::Tensor& hiddenStates,
    const torch::Tensor& value,
    const torch::Tensor& norm1Weight,
    const torch::Tensor& norm2Weight,
    const torch::Tensor& convNormWeight,
    int64_t hcMult,
    int64_t hiddenSize)
{
    TORCH_CHECK(
        keys.dim() == 4,
        "keys must have shape [B,L,HC,D].");

    TORCH_CHECK(
        hiddenStates.sizes()
            == keys.sizes(),
        "hidden_states shape must match keys.");

    TORCH_CHECK(
        keys.size(2) == hcMult,
        "HC must equal hc_mult.");

    TORCH_CHECK(
        keys.size(3) == hiddenSize,
        "D must equal hidden_size.");

    TORCH_CHECK(
        value.dim() == 3,
        "value must have shape [B,L,D].");

    TORCH_CHECK(
        value.size(0) == keys.size(0) &&
        value.size(1) == keys.size(1) &&
        value.size(2) == hiddenSize,
        "value shape mismatch.");

    TORCH_CHECK(
        norm1Weight.dim() == 2 &&
        norm1Weight.size(0) == hcMult &&
        norm1Weight.size(1) == hiddenSize,
        "norm1_weight shape mismatch.");

    TORCH_CHECK(
        norm2Weight.sizes()
            == norm1Weight.sizes(),
        "norm2_weight shape mismatch.");

    TORCH_CHECK(
        convNormWeight.sizes()
            == norm1Weight.sizes(),
        "conv_norm_weight shape mismatch.");

    TORCH_CHECK(
        keys.scalar_type()
            == torch::kBFloat16,
        "keys must be bfloat16.");

    TORCH_CHECK(
        hiddenStates.scalar_type()
            == torch::kBFloat16,
        "hidden_states must be bfloat16.");

    TORCH_CHECK(
        value.scalar_type()
            == torch::kBFloat16,
        "value must be bfloat16.");

    TORCH_CHECK(
        norm1Weight.scalar_type()
            == torch::kFloat32 &&
        norm2Weight.scalar_type()
            == torch::kFloat32 &&
        convNormWeight.scalar_type()
            == torch::kFloat32,
        "normalization weights must be float32.");
}

std::tuple<
    torch::Tensor,
    torch::Tensor>
engram_gate_stage_a_meta(
    const torch::Tensor& keys,
    const torch::Tensor& hiddenStates,
    const torch::Tensor& value,
    const torch::Tensor& norm1Weight,
    const torch::Tensor& norm2Weight,
    const torch::Tensor& convNormWeight,
    int64_t hcMult,
    int64_t hiddenSize,
    double normEps)
{
    check_stage_a_inputs(
        keys,
        hiddenStates,
        value,
        norm1Weight,
        norm2Weight,
        convNormWeight,
        hcMult,
        hiddenSize);

    auto fp32Options =
        keys.options()
            .dtype(torch::kFloat32);

    auto valueGated =
        torch::empty(
            keys.sizes(),
            fp32Options);

    auto normedValueGated =
        torch::empty(
            keys.sizes(),
            fp32Options);

    return std::make_tuple(
        valueGated,
        normedValueGated);
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl(
        "engram_gate_stage_a",
        engram_gate_stage_a_meta);
}

std::tuple<
    torch::Tensor,
    torch::Tensor>
engram_gate_stage_a_npu(
    const torch::Tensor& keys,
    const torch::Tensor& hiddenStates,
    const torch::Tensor& value,
    const torch::Tensor& norm1Weight,
    const torch::Tensor& norm2Weight,
    const torch::Tensor& convNormWeight,
    int64_t hcMult,
    int64_t hiddenSize,
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
        convNormWeight.is_contiguous(),
        "Stage A requires contiguous inputs.");

    check_stage_a_inputs(
        keys,
        hiddenStates,
        value,
        norm1Weight,
        norm2Weight,
        convNormWeight,
        hcMult,
        hiddenSize);

    auto outputs =
        engram_gate_stage_a_meta(
            keys,
            hiddenStates,
            value,
            norm1Weight,
            norm2Weight,
            convNormWeight,
            hcMult,
            hiddenSize,
            normEps);

    auto valueGated =
        std::get<0>(outputs);

    auto normedValueGated =
        std::get<1>(outputs);

    const int64_t B =
        keys.size(0);

    const int64_t L =
        keys.size(1);

    const int64_t totalItems =
        B * L * hcMult;

    int64_t numBlocks;
    int64_t itemsPerBlock;

    std::tie(
        numBlocks,
        itemsPerBlock) =
        calc_engram_stage_a_tiling_params(
            totalItems);

    auto stream =
        c10_npu::getCurrentNPUStream()
            .stream(false);

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

    auto valueGatedPtr =
        (GM_ADDR)valueGated.data_ptr();

    auto normedValueGatedPtr =
        (GM_ADDR)normedValueGated.data_ptr();

    const float normEpsFloat =
        static_cast<float>(normEps);

    auto aclCall = [=]() -> int {
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
            numBlocks,
            itemsPerBlock,
            stream);

        return 0;
    };

    at_npu::native::OpCommand::RunOpApi(
        "AgentForgeEngramStageA",
        aclCall);

    return std::make_tuple(
        valueGated,
        normedValueGated);
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl(
        "engram_gate_stage_a",
        engram_gate_stage_a_npu);
}

} // namespace cann_bench
