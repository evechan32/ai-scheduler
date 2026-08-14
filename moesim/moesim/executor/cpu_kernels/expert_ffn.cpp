#include <torch/extension.h>
#include <ATen/ATen.h>

// FP16 expert FFN executed on CPU: GELU(x @ w1^T) @ w2^T
torch::Tensor expert_ffn(torch::Tensor x, torch::Tensor w1, torch::Tensor w2) {
  TORCH_CHECK(x.is_cpu(), "expert_ffn requires CPU input");
  auto h = at::matmul(x, w1.transpose(0, 1));
  h = at::gelu(h);
  return at::matmul(h, w2.transpose(0, 1));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("expert_ffn", &expert_ffn, "FP16 expert FFN on CPU");
}
