#include "DataType.hpp"

void relu (
    MyDataType* input,
    int input_row,
    int input_col,
    int channel_size,
    MyDataType* output
) {
    for (int i = 0; i < input_row * input_col * channel_size; i++){
        if (input[i] > 0) {
            output[i] = input[i];
        } else {
            output[i] = 0;
        }
    }
}

MyDataType max_4 (MyDataType a, MyDataType b, MyDataType c, MyDataType d) {
    MyDataType result1 = a > b ? a : b;
    MyDataType result2 = c > d ? c : d;
    return result1 > result2 ? result1 : result2;
}

// kernel size = 2
// stride = 2
void maxpool (
    MyDataType* input,
    int input_row,
    int input_col,
    int channel_size,
    MyDataType* output
) {

    int output_row = input_row / 2;
    int output_col = input_col / 2;

    for (int i = 0; i < channel_size; i++) {
        for(int j = 0; j < output_row; j++){
            for(int k = 0; k < output_col; k++){
                int first = i * input_row * input_col + j * 2 * input_row + k * 2;
                int second = i * input_row * input_col + j * 2 * input_row + k * 2 + 1;
                int third = i * input_row * input_col + (j * 2 + 1) * input_row + k * 2;
                int fourth = i * input_row * input_col + (j * 2 + 1) * input_row + k * 2 + 1;
                output[i * output_row * output_col + j * output_row + k] = max_4(input[first], input[second], input[third], input[fourth]);
            }
        }
    }
}

int main () {
    return 0;
}