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
// Author: Zhen Zheng
// To be used in the future as a tool to generating the FP6 matrix from the FP16 matrix.

#include<iostream>

/*
 * Function to pack 4 fake quantized FP16 value into continuously stored 4 FP6 values.
 */
void print_binary(unsigned int num) {
    char binary[33];
    binary[32] = '\0';
    for (int i = 31; i >= 0; i--) {
        binary[i] = (num & 1) + '0';
        num >>= 1;
    }
    printf("%s\n", binary);
}


void cast_fp16_fp6(uint16_t* FP16x4, uint8_t* FP6x4, bool isRn)
{
    // Constants for FP6
    constexpr int exponent_nbits_fp6 = 3;
    constexpr int mantissa_nbits_fp6 = 2;
    constexpr int exp_bias_fp6 = (1 << (exponent_nbits_fp6 - 1)) - 1;
    // Constants for FP16
    constexpr int exponent_nbits_fp16 = 5;
    constexpr int mantissa_nbits_fp16 = 10;
    constexpr int exp_bias_fp16 = (1 << (exponent_nbits_fp16 - 1)) - 1;

    int fp6_temp[4];

    for(int i = 0; i < 4; i++) {
        fp6_temp[i] = 0;
        uint16_t source = FP16x4[i];
        int source_promote = int(source);
        int sign_bit = (source_promote >> 15);
        int exp_bit = (source_promote & 0x00007FFF) >> mantissa_nbits_fp16;
        int mant_bit = source_promote & ((1 << mantissa_nbits_fp16) - 1);
        int tail = mant_bit&0x000000FF;
        mant_bit = (mant_bit & 0x00000300) >> 8;
        if (isRn) {
            if (mant_bit == 0x00000003) {
                if (tail > 0x0000080) {
                    if((exp_bit == 0x00000000) || (exp_bit == 0x00000007)) {
                        // 无法进位了
                    } else {
                        exp_bit += 1; // 进位
                        mant_bit = 0x00000000; //进位后清0
                        // printf("case 1\n");
                        // print_binary(source);
                        // print_binary(sign_bit);
                        // print_binary(exp_bit);
                        // print_binary(mant_bit);
                    }
                }
            } else {
                if (tail > 0x0000080) {
                    mant_bit = mant_bit + 1;
                    // printf("case 2\n");
                    // print_binary(source);
                    // print_binary(sign_bit);
                    // print_binary(exp_bit);
                    // print_binary(mant_bit);
                }
            }
        }

        fp6_temp[i] = (sign_bit << (exponent_nbits_fp6 + mantissa_nbits_fp6)) |
                (exp_bit << mantissa_nbits_fp6) | mant_bit;
    }
    // Pack the values
    FP6x4[0] = fp6_temp[0] << 2 | (fp6_temp[1] >> 4);
    FP6x4[1] = (fp6_temp[1] & 0x0F) << 4 | (fp6_temp[2] >> 2);
    FP6x4[2] = (fp6_temp[2] & 0x03) << 6 | fp6_temp[3];
}

/*
 *  Function to prepack FP16 weights into continuous FP6 values.
 *
 *  Parameters:
 *     weight_16bit: input weight in FP16, size M*K
 *     weight_6bit: output weight in packed FP6, continuously stored, size M*K*6/8
 *     M, K: the shape of the weight
 */
void weight_prepacking_fp16_to_fp6(uint16_t* weight_16bit,
                                   uint8_t* weight_6bit_packed,
                                   size_t M,
                                   size_t K,
                                   bool isRn)
{
    // Every four 16-bit elements are packed into three 6-bit values (4*6bit == 3*8bit).
    if (K * 6 % 8 != 0) { throw std::invalid_argument("(K * 6 % 8) should be 0"); }
    size_t K_fp6_packed = K * 6 / 8;
    // #pragma omp parallel for
    for (auto m = 0; m < M; m++) {
        uint8_t* ptr_6bit = weight_6bit_packed + m * K_fp6_packed;
        uint16_t* ptr_16bit = weight_16bit + m * K;
        for (auto k = 0; k < K; k += 4) {
            cast_fp16_fp6(ptr_16bit, ptr_6bit, isRn);
            ptr_16bit += 4;
            ptr_6bit += 3;
        }
    }
}