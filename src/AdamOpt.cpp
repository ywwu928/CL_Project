// Adam reference: https://arxiv.org/pdf/1412.6980.pdf
// maybe: https://arxiv.org/pdf/2004.14180.pdf

#include "AdamOpt.hpp"
#include <cmath>
#include <iostream>
#include <exception>

template <class T> 
void AdamOpt<T>::update(std::vector<T> &current, const std::vector<T> grad) {

    if (current.size() != grad.size()) {
        throw std::invalid_argument("gradient vector must have same length as update vector");
    }

    // calculate mean gradient
    T avg_grad(0.0f);
    for (auto & elem : grad) avg_grad = avg_grad + elem;
    avg_grad = avg_grad / grad.size();

    // update m and v
    this->m = this->beta1 * this->m + this->beta1_inv * double(avg_grad);
    this->v = this->beta2 * this->v + this->beta2_inv * double(avg_grad * avg_grad);

    double alpha1 = this->alpha * sqrt(1 - this->beta2t) / (1 - this->beta1t);
    this->beta1t *= this->beta1;
    this->beta2t *= this->beta2;
    double update = alpha1 * m / (sqrt(v) + this->epsilon);

    for (auto & elem : current) elem = elem - T(update);
}

int main() {

    AdamOpt<double> adam;
    std::vector<double> weight(2, 1.0);
    std::vector<double> gradient(2, 1.0);
    adam.update(weight, gradient);
    adam.update(weight, gradient);
    adam.update(weight, gradient);
    for (auto & elem : weight) std::cout << elem << std::endl;

    return 0;
};
