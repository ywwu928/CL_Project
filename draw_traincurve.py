import matplotlib.pyplot as plt
import numpy as np



filename = 'result.txt'

loss = []
train = []
test = []
with open(filename, 'r') as f:
    args = f.readline()
    loss = f.readline()
    train = f.readline()
    test = f.readline()

print(args)
print(loss)
print(train)
print(test)


e = args.split('exponent')[1].split(',')[0].strip().split()[1]
m = args.split('mantissa')[1].split(',')[0].strip().split()[1].split('}')[0]
ofilename = e + '_' + m + '.png'

loss = loss.split('[')[1].split(']')[0].split(', ')
train = train.split('[')[1].split(']')[0].split(', ')
test = test.split('[')[1].split(']')[0].split(', ')
loss = [float(i) for i in loss]
train = [100.*float(i) for i in train]
test = [100.*float(i) for i in test]

x = np.arange(0, len(loss), 1, dtype=int)
print(x)

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(x, loss, 'r-', label='loss')
ax2.plot(x, train, 'b-', label='train')
ax2.plot(x, test, 'g-', label='test')
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss')
ax2.set_ylabel('accuracy')

plt.title('(1, ' + e + ', ' + m + ')')


# plt.ylabel('loss')
plt.legend()
plt.savefig(ofilename)
# plt.show()
# plt.close()