#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/mhc_sinkhorn_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def(
        "mhc_sinkhorn("
        "Tensor comb, "
        "int iter_step=20, "
        "float eps=1e-6"
        ") -> Tensor");
}

torch::Tensor mhc_sinkhorn_meta(
    const torch::Tensor& comb,
    int64_t iterStep,
    double eps)
{
    TORCH_CHECK(
        comb.dim() == 3,
        "MhcSinkhorn input must have shape "
        "[B, hc_mult, hc_mult].");

    TORCH_CHECK(
        comb.size(1) == comb.size(2),
        "MhcSinkhorn inner dimensions "
        "must form a square matrix.");

    TORCH_CHECK(
        comb.size(1) >= 2 &&
        comb.size(1) <= 16,
        "MhcSinkhorn hc_mult must be "
        "in [2, 16].");

    TORCH_CHECK(
        iterStep >= 1 &&
        iterStep <= 40,
        "MhcSinkhorn iter_step must be "
        "in [1, 40].");

    TORCH_CHECK(
        comb.scalar_type()
            == torch::kFloat32,
        "MhcSinkhorn supports "
        "float32 only.");

    return torch::empty_like(comb);
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl(
        "mhc_sinkhorn",
        mhc_sinkhorn_meta);
}

torch::Tensor mhc_sinkhorn_npu(
    const torch::Tensor& comb,
    int64_t iterStep,
    double eps)
{
    const c10::OptionalDeviceGuard guard(
        comb.device());

    TORCH_CHECK(
        comb.is_contiguous(),
        "MhcSinkhorn baseline requires "
        "contiguous input.");

    auto output =
        mhc_sinkhorn_meta(
            comb,
            iterStep,
            eps);

    const int64_t batchSize =
        comb.size(0);

    const int64_t hcMult =
        comb.size(1);

    int64_t numBlocks;
    int64_t matricesPerBlock;

    std::tie(
        numBlocks,
        matricesPerBlock) =
        calc_mhc_sinkhorn_tiling_params(
            batchSize);

    auto stream =
        c10_npu::getCurrentNPUStream()
            .stream(false);

    auto inputPtr =
        (GM_ADDR)comb.data_ptr();

    auto outputPtr =
        (GM_ADDR)output.data_ptr();

    const float epsFloat =
        static_cast<float>(eps);

    auto aclCall = [=]() -> int {
        MHC_SINKHORN_LAUNCH_FLOAT(
            inputPtr,
            outputPtr,
            batchSize,
            hcMult,
            iterStep,
            epsFloat,
            numBlocks,
            matricesPerBlock,
            stream);

        return 0;
    };

    at_npu::native::OpCommand::RunOpApi(
        "AgentForgeMhcSinkhorn",
        aclCall);

    return output;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl(
        "mhc_sinkhorn",
        mhc_sinkhorn_npu);
}

} // namespace cann_bench
