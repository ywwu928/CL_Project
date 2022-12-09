
#include "include/loss.hpp"
#include <cmath>
#include <iostream>

template <class T>
T CrossEntropyLoss<T>::forward(T* pred, T* target) {
    T loss = 0.0;
    for (int i = 0; i < this->shape[0]; i++) {
        loss += -target[i] * log(pred[i]);
    }
    return loss;

}

template <class T>
T* CrossEntropyLoss<T>::backward(T* pred, T* target) {
    T* grad = new T[this->shape[0]];
    for (int i = 0; i < this->shape[0]; i++) {
        grad[i] = -target[i] / pred[i];
    }
    return grad;
}

// int main(){

//     double pred[3] = {0.1, 0.2, 0.7};
//     double target[3] = {0.0, 0.0, 1.0};

//     int* shape = new int[1];
//     shape[0] = 3;

//     CrossEntropyLoss<double> loss(shape);
//     double l = loss.forward(pred, target);
//     auto g = loss.backward(pred, target);

//     std::cout << l << std::endl;
//     std::cout << g[0] << " " << g[1] << " " << g[2] << std::endl;
//     return 0;

// }
