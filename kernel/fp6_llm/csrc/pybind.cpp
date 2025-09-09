/***************************************************************************
 * Copyright 2023 The FLash-LLM Authors. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ***************************************************************************/
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "fp6_linear.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("linear_forward_cuda", &fp6_linear_forward_cuda, "Computes FP6-FP16 GEMM.");
    m.def("weight_prepacking_cpu", &weight_matrix_prepacking_cpu, "Weight prepacking.");
    m.def("weight_dequant_cpu", &weight_matrix_dequant_cpu, "Dequantize weight from fp6 to fp16.");
    m.def("weight_quant_to_fp6", &weight_quant_to_fp6, "weight quant and prepacking to fp6");
}