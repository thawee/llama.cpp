#include <initializer_list>
#include "ggml.h"
#include <cstdlib>
#include <cassert>
static constexpr int64_t GGML_METAL_MMA_ROWS_MAX = 16;
static int64_t ggml_metal_mul_mv_mma_rows_min(enum ggml_type type) {
    const char * name = type == GGML_TYPE_IQ3_S ? "GGML_METAL_MMA_IQ3_MIN_ROWS" :
                       type == GGML_TYPE_BF16  ? "GGML_METAL_MMA_BF16_MIN_ROWS" : nullptr;
    if (name) {
        const char * value = std::getenv(name);
        if (value) {
            char * end = nullptr;
            const long rows = std::strtol(value, &end, 10);
            if (end != value && *end == '\0' && rows >= 1 && rows <= GGML_METAL_MMA_ROWS_MAX + 1) {
                return rows;
            }
        }
    }
    switch (type) {
        case GGML_TYPE_F32:
            return 6;
        case GGML_TYPE_TQ2_0:
            return 5;
        case GGML_TYPE_BF16:
            return 4;
        case GGML_TYPE_F16:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
            return 3;
        default:
            return 2;
    }
}

int main() {
 unsetenv("GGML_METAL_MMA_IQ3_MIN_ROWS"); unsetenv("GGML_METAL_MMA_BF16_MIN_ROWS");
 assert(ggml_metal_mul_mv_mma_rows_min(GGML_TYPE_IQ3_S)==2);
 assert(ggml_metal_mul_mv_mma_rows_min(GGML_TYPE_BF16)==4);
 for (const char * bad : {"0", "18", "garbage", "3oops", "9999999999999999999999"}) {
  setenv("GGML_METAL_MMA_IQ3_MIN_ROWS",bad,1);
  assert(ggml_metal_mul_mv_mma_rows_min(GGML_TYPE_IQ3_S)==2);
 }
 setenv("GGML_METAL_MMA_IQ3_MIN_ROWS","1",1);
 setenv("GGML_METAL_MMA_BF16_MIN_ROWS","17",1);
 assert(ggml_metal_mul_mv_mma_rows_min(GGML_TYPE_IQ3_S)==1);
 assert(ggml_metal_mul_mv_mma_rows_min(GGML_TYPE_BF16)==17);
 assert(ggml_metal_mul_mv_mma_rows_min(GGML_TYPE_Q4_K)==3);
}
