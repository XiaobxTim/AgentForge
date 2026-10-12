#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/swi_glu_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def(
        "swi_glu(Tensor input, int dim=-1) -> Tensor");
}

static int64_t normalize_dim(
    int64_t dim,
    int64_t ndim)
{
    if (dim < 0) {
        dim += ndim;
    }

    TORCH_CHECK(
        dim >= 0 && dim < ndim,
        "SwiGLU dim out of range.");

    return dim;
}

torch::Tensor swi_glu_meta(
    const torch::Tensor& input,
    int64_t dim)
{
    TORCH_CHECK(
        input.dim() > 0,
        "SwiGLU input must have rank >= 1.");

    dim =
        normalize_dim(dim, input.dim());

    auto outputSizes =
        input.sizes().vec();

    TORCH_CHECK(
        outputSizes[dim] % 2 == 0,
        "SwiGLU split dimension must be even.");

    outputSizes[dim] /= 2;

    return torch::empty(
        outputSizes,
        input.options());
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl(
        "swi_glu",
        swi_glu_meta);
}

torch::Tensor swi_glu_npu(
    const torch::Tensor& input,
    int64_t dim)
{
    const c10::OptionalDeviceGuard guard(
        input.device());

    TORCH_CHECK(
        input.is_contiguous(),
        "SwiGLU baseline requires contiguous input.");

    dim =
        normalize_dim(dim, input.dim());

    const int64_t dimSize =
        input.size(dim);

    TORCH_CHECK(
        dimSize % 2 == 0,
        "SwiGLU split dimension must be even.");

    auto output =
        swi_glu_meta(input, dim);

    const int64_t halfDim =
        dimSize / 2;

    int64_t innerLength = 1;

    for (int64_t i = dim + 1;
         i < input.dim();
         ++i) {
        innerLength *= input.size(i);
    }

    const int64_t groupLength =
        halfDim * innerLength;

    const int64_t totalOutputLength =
        input.numel() / 2;

    int64_t numBlocks;
    int64_t blockLength;
    uint32_t tileElements;

    std::tie(
        numBlocks,
        blockLength,
        tileElements) =
        calc_swi_glu_tiling_params(
            totalOutputLength);

    auto stream =
        c10_npu::getCurrentNPUStream()
            .stream(false);

    auto inputPtr =
        (GM_ADDR)input.data_ptr();

    auto outputPtr =
        (GM_ADDR)output.data_ptr();

    auto aclCall = [=]() -> int {
        if (input.scalar_type()
            == torch::kFloat32) {
            SWI_GLU_LAUNCH_FLOAT(
                inputPtr,
                outputPtr,
                totalOutputLength,
                groupLength,
                numBlocks,
                blockLength,
                tileElements,
                stream);

        } else if (input.scalar_type()
                   == torch::kFloat16) {
            SWI_GLU_LAUNCH_HALF(
                inputPtr,
                outputPtr,
                totalOutputLength,
                groupLength,
                numBlocks,
                blockLength,
                tileElements,
                stream);

        } else if (input.scalar_type()
                   == torch::kBFloat16) {
            SWI_GLU_LAUNCH_BFLOAT16(
                inputPtr,
                outputPtr,
                totalOutputLength,
                groupLength,
                numBlocks,
                blockLength,
                tileElements,
                stream);

        } else {
            TORCH_CHECK(
                false,
                "SwiGLU supports float16, "
                "float32 and bfloat16 only.");
        }

        return 0;
    };

    at_npu::native::OpCommand::RunOpApi(
        "AgentForgeSwiGlu",
        aclCall);

    return output;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl(
        "swi_glu",
        swi_glu_npu);
}

} // namespace cann_bench
