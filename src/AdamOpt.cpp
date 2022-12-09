// Adam reference: https://arxiv.org/pdf/1412.6980.pdf
// maybe: https://arxiv.org/pdf/2004.14180.pdf

#include "AdamOpt.hpp"
#include <cmath>
#include <iostream>
#include <exception>

// for testing
#include <random>

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

// std::random_device rd;
// std::mt19937 gen(rd());
// std::uniform_real_distribution<> dis(-10.0, 10.0);

// double get_gradient(double x) {
//     return 2 * x + 0.1 * dis(gen);
// }

// int main() {

//     const int iterations = 200;
//     const int epochs = 10;
//     const int n_weights = 4;


//     AdamOpt<double> adam(0.1);
//     std::vector<double> weight(n_weights);
//     std::vector<double> gradient(n_weights);

//     int x = epochs;
//     while (x --> 0) {
//         double initial = dis(gen);
//         for (size_t i = 0; i < weight.size(); i++) weight[i] = initial;

//         int y = iterations;
//         while (y --> 0) {
//             for (size_t i = 0; i < weight.size(); i++) gradient[i] = get_gradient(weight[i]);
//             adam.update(weight, gradient);
//             // for (auto & elem : weight) std::cout << elem << " ";
//             // std::cout << std::endl;
//             double avg_weight = 0;
//             for (auto e : weight) avg_weight += e;
//             for (auto & e : weight) e = avg_weight / n_weights;
//         }
//         for (auto & elem : weight) std::cout << elem << " ";
//         std::cout << std::endl;
//     }

//     return 0;
// };
