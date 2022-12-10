import argparse
from argparse import ArgumentParser
from SimpleNet import SimpleNet, NetRunner

def main():
    '''
    configurable arguments:
    -----------------------
    epochs          --epochs        #epochs
    learning rate   --lr            #learning rate
    cuda            --cuda
    exponent        --exponent, -e  #exponent bits
    mantissa        --mantissa, -m  #mantissa bits


    example usage: (5b exponent, 10b mantissa, use cuda)
    python3 main.py -e 5 -m 10 --cuda
    '''

    parser = ArgumentParser(description='CNN with floating point custom exponent and mantissa widths')
    parser.add_argument('--exponent', '-e', dest='exponent', type=int)
    parser.add_argument('--mantissa', '-m', dest='mantissa', type=int)
    parser.add_argument('--epochs', dest='epochs', type=int)
    parser.add_argument('--lr', dest='lr', type=float)
    parser.add_argument('--cuda', dest='enable_cuda', action='store_true') 
    args = {key:val for key,val in vars(parser.parse_args()).items() if val is not None}

    NetRunner(args).run()

if __name__ == '__main__':
    main()
    print('Finished Training') 