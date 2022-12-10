// cross entropy loss function  

#ifndef LOSS_HPP
#define LOSS_HPP

template <class T>
class Loss {
public:
    Loss() {}
    virtual ~Loss() {}
    virtual T forward(T* pred, T* target) = 0;
    virtual T* backward(T* pred, T* target) = 0;
};

template <class T>
class CrossEntropyLoss : public Loss<T> {
public:
    CrossEntropyLoss(int* shape) : Loss<T>(), shape(shape) {}
    ~CrossEntropyLoss() {}
    T forward(T* pred, T* target);
    T* backward(T* pred, T* target);
private:
    int* shape;
};


#endif // LOSS_HPP

