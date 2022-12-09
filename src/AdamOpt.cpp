// Adam reference: https://arxiv.org/pdf/1412.6980.pdf
// maybe: https://arxiv.org/pdf/2004.14180.pdf

#include "AdamOpt.hpp"
#include <cmath>
#include <iostream>

template <class T> 
T AdamOpt<T>::update(T current, T grad) {
    // update m and v
    this->m = this->beta1 * this->m + this->beta1_inv * double(grad);
    this->v = this->beta2 * this->v + this->beta2_inv * double(grad * grad);

    double alpha1 = this->alpha * sqrt(1 - this->beta2t) / (1 - this->beta1t);
    this->beta1t *= this->beta1;
    this->beta2t *= this->beta2;
    return current - T(alpha1 * m / (sqrt(v) + this->epsilon));
}

int main() {

    AdamOpt<double> adam;
    double u = adam.update(1.0, 1.0);
    std::cout << u << std::endl;

    return 0;
};
