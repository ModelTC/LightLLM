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
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

void DeQuantMatrix_FP6_To_FP16(half* A_16bit_h, unsigned char* A_6bit_h, size_t M, size_t K, half* scale) {
    assert(M%64==0);                 // Currently, M must be a multiple of 64.
    assert(K%64==0);                 // Currently, K must be a multiple of 64.
    size_t TotalSizeInByte = M*K*6/8;
    //
    half* OutPTR = A_16bit_h;
    for(size_t i=0; i<TotalSizeInByte/3; i++) {    // 4 FP6 = 3 Bytes for each Loop
        unsigned char   B1  = A_6bit_h[i*3+0] & 0xfc;
                        B1  = (B1&0x80) | ((B1>>2)&0x1f);
        unsigned char   B2  = (A_6bit_h[i*3+0]<<6) | ((A_6bit_h[i*3+1]>>2)&0xfc);
                        B2  = (B2&0x80) | ((B2>>2)&0x1f);
        unsigned char   B3  = (A_6bit_h[i*3+1]<<4) | ((A_6bit_h[i*3+2]>>4)&0xfc);
                        B3  = (B3&0x80) | ((B3>>2)&0x1f);
        unsigned char   B4  = A_6bit_h[i*3+2]<<2;
                        B4  = (B4&0x80) | ((B4>>2)&0x1f);
        half            FP1, FP2, FP3, FP4;
        unsigned char   *PTR1, *PTR2, *PTR3, *PTR4;
        PTR1 = reinterpret_cast<unsigned char*>(&FP1);
        PTR2 = reinterpret_cast<unsigned char*>(&FP2);
        PTR3 = reinterpret_cast<unsigned char*>(&FP3);
        PTR4 = reinterpret_cast<unsigned char*>(&FP4);
        PTR1[0] = 0;    PTR1[1] = B1;   // small endian for X86 CPU
        PTR2[0] = 0;    PTR2[1] = B2;
        PTR3[0] = 0;    PTR3[1] = B3;
        PTR4[0] = 0;    PTR4[1] = B4;
        OutPTR[0] = __float2half_rn ( __half2float(FP1) * 4096.0f * __half2float(scale[(4*i)/K]) );
        OutPTR[1] = __float2half_rn ( __half2float(FP2) * 4096.0f * __half2float(scale[(4*i)/K]) );
        OutPTR[2] = __float2half_rn ( __half2float(FP3) * 4096.0f * __half2float(scale[(4*i)/K]) );
        OutPTR[3] = __float2half_rn ( __half2float(FP4) * 4096.0f * __half2float(scale[(4*i)/K]) );
        //
        OutPTR +=4;
    }
}