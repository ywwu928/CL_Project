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

void fc (
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

void max_pool (
    MyDataType* input,
    int input_row,
    int input_col,
    int channel_size,
    MyDataType* output
);