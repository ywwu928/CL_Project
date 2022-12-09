#include "include/Layers.hpp"

/* Forward functions */

// padding = 1
void zero_pad (
    MyDataType* input,
    int input_row,
    int input_col,
    int channel_size,
    MyDataType* output
) {
    int output_row = input_row + 2;
    int output_col = input_col + 2;

    for (int i = 0; i < output_row * output_col * channel_size; i++) {
        output[i] = 0;
    }

    for(int i = 0; i < channel_size; i++){
        for(int j = 0; j < input_row; j++){
            for(int k = 0; k < input_col; k++){
                output[i * output_row * output_col + (j + 1) * output_col + (k + 1)] = input[i * input_row * input_col + j * input_col + k];
            }
        }
    }
}

void im_to_col (
    MyDataType* input,
    int i_mat_row,
    int i_mat_col,
    int o_mat_row,
    int o_mat_col,
    int i_channel_size,
    int filter_row,
    int filter_col,
    MyDataType* output
) {
    for(int i = 0; i < o_mat_row; i++){
        for(int j = 0; j < o_mat_col; j++){
            for(int k = 0; k < i_channel_size; k++){
                for(int l = 0; l < filter_row; l++){
                    for(int m = 0; m < filter_col; m++){
                        output[i * o_mat_col * i_channel_size * filter_row * filter_col + j * i_channel_size * filter_row * filter_col + k * filter_row * filter_col + l * filter_col + m] = input[k * i_mat_row * i_mat_col + (i + l) * i_mat_col + (j + m)];
                    }
                }
            }
        }
    }
}

void reorder_filter (
    MyDataType* input,
    int input_row,
    int input_col,
    int input_channel,
    int output_channel,
    MyDataType* output
) {
    for(int i = 0; i < input_channel; i++){
        for(int j = 0; j < input_row; j++){
            for(int k = 0; k < input_col; k++){
                for(int l = 0; l < output_channel; l++){
                    output[i * input_row * input_col * output_channel + j * input_col * output_channel + k * output_channel + l] = input[l * input_channel * input_row * input_col + i * input_row * input_col + j * input_col + k];
                }
            }
        }
    }
}

void mat_mul (
    MyDataType* mat_a,
    MyDataType* mat_b,
    int rows_a,
    int cols_a,
    int cols_b,
    MyDataType* output
) {
    for (int i = 0; i < rows_a; i++) {
        for (int j = 0; j < cols_b; j++) {
            output[i * cols_b + j] = 0;
            for (int k = 0; k < cols_a; k++) {
                output[i * cols_b + j] = output[i * cols_b + j] + mat_a[i * cols_a + k] * mat_b[k * cols_b + j];
            }
        }
    }
}

void reorder_image (
    MyDataType* input,
    int input_row,
    int input_col,
    int channel_size,
    MyDataType* output
) {
    for(int i = 0; i < channel_size; i++){
        for(int j = 0; j < input_row; j++){
            for(int k = 0; k < input_col; k++){
                output[i * input_row * input_col + j * input_col + k] = input[j * input_col * channel_size + k * channel_size + i];
            }
        }
    }
}

void add_bias (
    MyDataType* input,
    MyDataType* bias,
    int input_row,
    int input_col,
    int channel_size,
    MyDataType* output
) {
    for(int i = 0; i < channel_size; i++){
        for(int j = 0; j < input_row; j++){
            for(int k = 0; k < input_col; k++){
                output[i * input_row * input_col + j * input_col + k] = input[i * input_row * input_col + j * input_col + k] + bias[i];
            }
        }
    }
}

void conv (
    MyDataType* input,
    MyDataType* filter,
    MyDataType* bias,
    int input_row,
    int input_col,
    int input_channel,
    int filter_size,
    int output_channel,
    MyDataType* output
) {
    // zero padding
    MyDataType input_padded [(input_row + 2) * (input_col + 2) * input_channel];
    zero_pad(input, input_row, input_col, input_channel, input_padded);

    // create IFmap matrix
    MyDataType ifmap_matrix [(1 * input_row * input_col) * (input_channel * filter_size * filter_size)];
    im_to_col(input_padded, input_row + 2, input_col + 2, input_row, input_col, input_channel, filter_size, filter_size, ifmap_matrix);

    // create filter matrix
    MyDataType filter_matrix [(input_channel * filter_size * filter_size) * output_channel];
    reorder_filter(filter, filter_size, filter_size, input_channel, output_channel, filter_matrix);

    // matrix multiplication
    MyDataType output_matrix [(1 * input_row * input_col) * output_channel];
    mat_mul(ifmap_matrix, filter_matrix, (1 * input_row * input_col), (input_channel * filter_size * filter_size), output_channel, output_matrix);

    // reorder output matrix
    MyDataType output_reordered [input_row * input_col * output_channel * 1];
    reorder_image(output_matrix, input_row, input_col, output_channel, output_reordered);

    // add bias
    add_bias(output_reordered, bias, input_row, input_col, output_channel, output);
}

void fc_forward (
    MyDataType* input,
    MyDataType* filter,
    MyDataType* bias,
    int input_row,
    int input_col,
    int input_channel,
    int filter_size,
    int output_channel,
    MyDataType* output
) {
    // create filter matrix
    MyDataType filter_matrix [(input_channel * filter_size * filter_size) * output_channel];
    reorder_filter(filter, filter_size, filter_size, input_channel, output_channel, filter_matrix);

    // matrix multiplication
    MyDataType output_matrix [output_channel];
    mat_mul(input, filter_matrix, 1, (input_row * input_col * input_channel), output_channel, output_matrix);

    // add bias
    add_bias(output_matrix, bias, 1, 1, output_channel, output);
}

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

int max_4_index (MyDataType a, MyDataType b, MyDataType c, MyDataType d) {
    MyDataType result1 = a > b ? a : b;
    int index1 = a > b ? 1 : 2;
    MyDataType result2 = c > d ? c : d;
    int index2 = c > d ? 3 : 4;
    return result1 > result2 ? index1 : index2;
}

// kernel size = 2
// stride = 2
void max_pool_forward (
    MyDataType* input,
    int input_row,
    int input_col,
    int channel_size,
    MyDataType* output,
    int* max_map
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
                
                int index = max_4_index(input[first], input[second], input[third], input[fourth]);
                switch(index) {
                    case 1:
                        max_map[first] = 1;
                        max_map[second] = 0;
                        max_map[third] = 0;
                        max_map[fourth] = 0;
                        break;
                    case 2:
                        max_map[first] = 0;
                        max_map[second] = 1;
                        max_map[third] = 0;
                        max_map[fourth] = 0;
                        break;
                    case 3:
                        max_map[first] = 0;
                        max_map[second] = 0;
                        max_map[third] = 1;
                        max_map[fourth] = 0;
                        break;
                    case 4:
                        max_map[first] = 0;
                        max_map[second] = 0;
                        max_map[third] = 0;
                        max_map[fourth] = 1;
                        break;
                    default:
                        break;
                }
            }
        }
    }
}

/* Backward functions */

void fc_backward (
    MyDataType* expected_curr,
    MyDataType* output_curr,
    MyDataType* weight_curr,
    int neuron_num_curr,
    MyDataType* delta_output_curr,
    MyDataType* delta_bias_curr,
    MyDataType* output_prev,
    int neuron_num_prev,
    MyDataType* delta_weight_prev,
    MyDataType* delta_output_prev
) {
    for (int i = 0; i < neuron_num_curr; i++) {
        delta_output_curr[i] = expected_curr[i] - output_curr[i];

        for (int j = 0; j < neuron_num_prev; j++) {
            delta_weight_prev[j * neuron_num_curr + i] = delta_output_curr[i] * output_prev[j];
            delta_output_prev[j] = weight_curr[i] * delta_output_curr[i];
        }

        delta_bias_curr[i] = delta_output_curr[i];
    }
}

// kernel size = 2
// stride = 2
void max_pool_backward (
    int* max_map,
    MyDataType* delta_output,
    int neuron_num
    // int input_row,
    // int input_col,
    // int channel_size
) {
    for (int i = 0; i < neuron_num; i++) {
        if (max_map[i] == 0) {
            delta_output[i] = 0;
        }
    }
    // for (int i = 0; i < channel_size; i++) {
    //     for(int j = 0; j < input_row; j++){
    //         for(int k = 0; k < input_col; k++){
    //             if (max_map[i * input_row * input_col + j * input_row + k] == 0) {
    //                 delta_output[i * input_row * input_col + j * input_row + k] = 0
    //             }
    //         }
    //     }
    // }
}

void relu_backward (
    MyDataType* input,
    MyDataType* delta_output,
    int neuron_num
) {
    for (int i = 0; i < neuron_num; i++) {
        if (input[i] < 0) {
            delta_output[i] = 0;
        }
    }
}

void conv_backward (
    MyDataType* weight_curr,
    MyDataType* delta_output_curr,
    int neuron_num_curr,
    MyDataType* delta_bias_curr,
    MyDataType* output_prev,
    int neuron_num_prev,
    MyDataType* delta_weight_prev,
    MyDataType* delta_output_prev
) {
    for (int i = 0; i < neuron_num_curr; i++) {
        for (int j = 0; j < neuron_num_prev; j++) {
            delta_weight_prev[j * neuron_num_curr + i] = delta_output_curr[i] * output_prev[j];
            delta_output_prev[j] = weight_curr[i] * delta_output_curr[i];
        }

        delta_bias_curr[i] = delta_output_curr[i];
    }
}

int main () {
    return 0;
}