import matplotlib.pyplot as plt
import numpy as np

files = ['batch_size_32_epochs_30_lr_0.1_enable_cuda_True_L1norm_False_simpleNet_True_activation_relu_train_curve_True_optimization_SGD_exponent_10_mantissa_1.txt',
'batch_size_32_epochs_30_lr_0.1_enable_cuda_True_L1norm_False_simpleNet_True_activation_relu_train_curve_True_optimization_SGD_exponent_10_mantissa_5.txt',
'batch_size_32_epochs_30_lr_0.1_enable_cuda_True_L1norm_False_simpleNet_True_activation_relu_train_curve_True_optimization_SGD_exponent_11_mantissa_6.txt',
'batch_size_32_epochs_30_lr_0.1_enable_cuda_True_L1norm_False_simpleNet_True_activation_relu_train_curve_True_optimization_SGD_exponent_12_mantissa_3.txt',
'batch_size_32_epochs_30_lr_0.1_enable_cuda_True_L1norm_False_simpleNet_True_activation_relu_train_curve_True_optimization_SGD_exponent_13_mantissa_2.txt',
'batch_size_32_epochs_30_lr_0.1_enable_cuda_True_L1norm_False_simpleNet_True_activation_relu_train_curve_True_optimization_SGD_exponent_14_mantissa_1.txt',
'batch_size_32_epochs_30_lr_0.1_enable_cuda_True_L1norm_False_simpleNet_True_activation_relu_train_curve_True_optimization_SGD_exponent_4_mantissa_3.txt',
'batch_size_32_epochs_30_lr_0.1_enable_cuda_True_L1norm_False_simpleNet_True_activation_relu_train_curve_True_optimization_SGD_exponent_5_mantissa_2.txt',
'batch_size_32_epochs_30_lr_0.1_enable_cuda_True_L1norm_False_simpleNet_True_activation_relu_train_curve_True_optimization_SGD_exponent_6_mantissa_1.txt',
'batch_size_32_epochs_30_lr_0.1_enable_cuda_True_L1norm_False_simpleNet_True_activation_relu_train_curve_True_optimization_SGD_exponent_6_mantissa_5.txt',
'batch_size_32_epochs_30_lr_0.1_enable_cuda_True_L1norm_False_simpleNet_True_activation_relu_train_curve_True_optimization_SGD_exponent_7_mantissa_4.txt',
'batch_size_32_epochs_30_lr_0.1_enable_cuda_True_L1norm_False_simpleNet_True_activation_relu_train_curve_True_optimization_SGD_exponent_8_mantissa_3.txt',
'batch_size_32_epochs_30_lr_0.1_enable_cuda_True_L1norm_False_simpleNet_True_activation_relu_train_curve_True_optimization_SGD_exponent_9_mantissa_2.txt']

for filename in files:
    filename = 'result/' + filename
    loss = []
    train = []
    test = []
    with open(filename, 'r') as f:
        args = f.readline()
        loss = f.readline()
        train = f.readline()
        test = f.readline()

    # print(args)
    # print(loss)
    # print(train)
    # print(test)


    e = args.split('exponent')[1].split(',')[0].strip().split()[1]
    m = args.split('mantissa')[1].split(',')[0].strip().split()[1].split('}')[0]
    ofilename = 'traincurve/' + e + '_' + m + '.png'

    loss = loss.split('[')[1].split(']')[0].split(', ')
    train = train.split('[')[1].split(']')[0].split(', ')
    test = test.split('[')[1].split(']')[0].split(', ')
    loss = [float(i) for i in loss]
    train = [100.*float(i) for i in train]
    test = [100.*float(i) for i in test]

    x = np.arange(0, len(loss), 1, dtype=int)
    # print(x)

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(x, loss, 'r-', label='loss')
    ax2.plot(x, train, 'b-', label='train')
    ax2.plot(x, test, 'g-', label='test')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax2.set_ylabel('accuracy')

    #ax1 y limit
    ax1.set_ylim([0, 1])

    plt.title('(1, ' + e + ', ' + m + ')')
    #lengend
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')


    # plt.ylabel('loss')
    plt.savefig(ofilename)
    # plt.show()
    # plt.close()