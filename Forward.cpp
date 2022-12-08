#include "DataType.hpp"

void reluinplace(float* a,
                int mat_row,
                int mat_col,
                int channel_size){
    _reluinplace_ :
    for (int i = 0; i < mat_row*mat_col*channel_size; i++){
        #pragma HLS LOOP_TRIPCOUNT min=12544 max=12544
        a[i] = relu_max_zero(a[i]);
    }
}