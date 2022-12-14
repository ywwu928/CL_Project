#include "DataType.hpp"

void zero_pad (
    MyDataType* input,
    int input_row,
    int input_col,
    int channel_size,
    MyDataType* output
);

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
);

void reorder_filter (
    MyDataType* input,
    int input_row,
    int input_col,
    int input_channel,
    int output_channel,
    MyDataType* output
);

void mat_mul (
    MyDataType* mat_a,
    MyDataType* mat_b,
    int rows_a,
    int cols_a,
    int cols_b,
    MyDataType* output
);

void reorder_image (
    MyDataType* input,
    int input_row,
    int input_col,
    int channel_size,
    MyDataType* output
);

void add_bias (
    MyDataType* input,
    MyDataType* bias,
    int input_row,
    int input_col,
    int channel_size,
    MyDataType* output
);

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
);

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
);

void relu (
    MyDataType* input,
    int input_row,
    int input_col,
    int channel_size,
    MyDataType* output
);

MyDataType max_4 (MyDataType a, MyDataType b, MyDataType c, MyDataType d);

int max_4_index (MyDataType a, MyDataType b, MyDataType c, MyDataType d);

void max_pool_forward (
    MyDataType* input,
    int input_row,
    int input_col,
    int channel_size,
    MyDataType* output,
    int* max_map
);

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
);

void max_pool_backward (
    MyDataType* max_map,
    MyDataType* delta_output,
    int neuron_num
    // int input_row,
    // int input_col,
    // int channel_size
);

void relu_backward (
    MyDataType* input,
    MyDataType* delta_output,
    int neuron_num
);

void conv_backward (
    MyDataType* weight_curr,
    MyDataType* delta_output_curr,
    int neuron_num_curr,
    MyDataType* delta_bias_curr,
    MyDataType* output_prev,
    int neuron_num_prev,
    MyDataType* delta_weight_prev,
    MyDataType* delta_output_prev
);
