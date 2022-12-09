#include "include/typedef.hpp"
#include <iostream>

double pow2(int exp){
    double result = 1;
    if(exp == 0) return 1;

    if(exp > 0){
        for(int i = 0; i < exp; i++){
            result *= 2;
        }
    }else{
        for(int i = 0; i < -exp; i++){
            result /= 2;
        }
    }
    return result;
}

bool isOverflow(myfloat a){
    int EXPMIN = - (1 << (a.getExponentBits()-1));
    int EXPMAX = (1 << (a.getExponentBits()-1)) - 1;
    int MANTMIN = 0;
    int MANTMAX = (1 << a.getMantissaBits()) - 1;

    // std::cout << "EXPMIN: " << EXPMIN << std::endl;
    // std::cout << "EXPMAX: " << EXPMAX << std::endl;
    // std::cout << "MANTMIN: " << MANTMIN << std::endl;
    // std::cout << "MANTMAX: " << MANTMAX << std::endl;

    if(a.getExponent() < EXPMIN || a.getExponent() > EXPMAX){
        return true;
    }
    if(a.getMantissa() < MANTMIN || a.getMantissa() > MANTMAX){
        return true;
    }
    return false;
}

void myfloat::normalize(){
    int mask = (1 << (this->getMantissaBits())) - 1;
    int mantissa = this->getMantissa();
    int exponent = this->getExponent();

    if(mantissa == 0){
        this->exponent = 0;
        this->mantissa = 0;
        this->value = 0;
        this->iszero = true;
        return;        
    }

    while(mantissa > mask){
        mantissa >>= 1;
        exponent++;
    }
    while(mantissa < (mask>>1)){
        mantissa <<= 1;
        exponent--;
    }
    // std::cout << "normal mast" << mask << std::endl;
    // std::cout << "normal man: " << mantissa << std::endl;
    // std::cout << "normal exp: " << exponent << std::endl;

    this->mantissa = mantissa;
    this->exponent = exponent;
    this->iszero = false;

}

void myfloat::setValue(float num){
    float value = num;
    int exponent = 0;
    int mantissa = 0;
    SIGN sign = POSITIVE;

    if(value == 0){
        this->setValue(POSITIVE, 0, 0);
        return;
    }


    if(value < 0){
        sign = NEGATIVE;
        value = -value;
    }
    long long *p = (long long*)&num;
    long long bit_rep = *p;
    
    exponent = ((bit_rep >> 23) & 0xff); // 125
    exponent -= 127;
    mantissa = bit_rep & 0x7fffff;
    mantissa = mantissa >> (24 - this->mantissa_bits);
    mantissa += (1 << (this->mantissa_bits-1));

    this->sign = sign;
    this->exponent = exponent;
    this->mantissa = mantissa;
    this->setValue(sign, mantissa, exponent);

}

void myfloat::setValue(SIGN sign, int mantissa, int exponent){
    double value = 0;
    if(mantissa == 0){
        this->iszero = true;
        this->value = 0;
        this->exponent = 0;
        this->mantissa = 0;
        return;
    }

    int man = mantissa & this->mantissa_mask;
    int exp = exponent;
    this->exponent = exp;
    this->mantissa = man;
    this->sign = sign;
    this->iszero = false;
    value = double(man) / double((1 << (this->mantissa_bits-1))) * pow2(exp);
    this->value = sign == POSITIVE ? value : -value;
    
    this->normalize();
    if(isOverflow(*this)){
        std::cout << "Overflow" << std::endl;
        exit(1);
    }
}

// // myfloat operator=(myfloat a, myfloat b){
//     myfloat result;
//     result.setMantissa(b.getMantissa());
//     result.setExponent(b.getExponent());

//     return result;
// }

myfloat operator+(myfloat& a, myfloat& b){
    myfloat result(a);

    if(a.getMantissaBits() != b.getMantissaBits()){
        std::cout << "Mantissa bits must be the same" << std::endl;
        exit(1);
    }

    if(a.getExponentBits() != b.getExponentBits()){
        std::cout << "Exponent bits must be the same" << std::endl;
        exit(1);
    }
    myfloat small;
    myfloat large;
    
    small = a > b ? b : a;
    large = a > b ? a : b;
    
    int exponent = large.getExponent();
    int man = small.getMantissa() >> (exponent - small.getExponent());

    int resultman = man + large.getMantissa();
    int resultexp = large.getExponent();

    SIGN sign = large.getSign();

    result.setValue(sign, resultman, resultexp); 
    return result;
}

myfloat operator-(myfloat& a, myfloat& b){
    myfloat result(a);
    if(a.getMantissaBits() != b.getMantissaBits()){
        std::cout << "Mantissa bits must be the same" << std::endl;
        exit(1);
    }


    if(a.getExponentBits() != b.getExponentBits()){
        std::cout << "Exponent bits must be the same" << std::endl;
        exit(1);
    }
    
    myfloat small;
    myfloat large;
    SIGN sign;
    if(a > b){
        small = b;
        large = a;
        sign = a.getSign();
    }
    else{
        small = a;
        large = b;
        sign = a.getSign() == POSITIVE ? NEGATIVE : POSITIVE;
    }

    int exponent = large.getExponent();
    int man = small.getMantissa() >> (exponent - small.getExponent());

    int resultman = large.getMantissa() - man;
    int resultexp = large.getExponent();
    

    result.setValue(sign, resultman, resultexp);
    return result;
}

myfloat operator*(myfloat& a, myfloat& b){
    myfloat result(a);
    if (a.isZero() || b.isZero())
        return result;

    if (a.getMantissaBits() != b.getMantissaBits()){
        std::cout << "Mantissa bits must be the same" << std::endl;
        exit(1);
    }

    if (a.getExponentBits() != b.getExponentBits()){
        std::cout << "Exponent bits must be the same" << std::endl;
        exit(1);
    }

    int exponent = a.getExponent() + b.getExponent();
    int mantissa = a.getMantissa() * b.getMantissa();
    int mask = (1 << a.getMantissaBits()) - 1;
    SIGN sign = a.getSign() == b.getSign() ? POSITIVE : NEGATIVE;
    mantissa = mantissa >> (a.getMantissaBits() - 1);

    if (mantissa > mask){
        exponent++;
        mantissa = mantissa >> 1;
    }

    if (mantissa < (mask >> 1)){
        exponent--;
        mantissa = mantissa << 1;
    }
    result.setValue(sign, mantissa, exponent);
    return result;

}

myfloat operator/(myfloat& a, myfloat& b){
    myfloat result(a);
    if (a.isZero() || b.isZero()){
        std::cout << "Divide by zero" << std::endl;
        exit(1);
    }

    if (a.getMantissaBits() != b.getMantissaBits()){
        std::cout << "Mantissa bits must be the same" << std::endl;
        exit(1);
    }
    if (a.getExponentBits() != b.getExponentBits()){
        std::cout << "Exponent bits must be the same" << std::endl;
        exit(1);
    }
    
    double value = a.getValue() / b.getValue();
    result.setValue(value);
    return result;
}


bool operator<(myfloat& a, myfloat& b){
    return a.getValue() < b.getValue();
}

bool operator>(myfloat& a, myfloat& b){
    return a.getValue() > b.getValue();
}

bool operator<=(myfloat& a, myfloat& b){
    return a.getValue() <= b.getValue();
}

bool operator>=(myfloat& a, myfloat& b){
    return a.getValue() >= b.getValue();
}

bool operator==(myfloat& a, myfloat& b){
    return a.getValue() == b.getValue();
}

std::ostream& operator<<(std::ostream& os, myfloat& a){
    os << a.getValue();
    return os;
}

void myfloat::print(){
    std::cout << this->getValue() << std::endl;
}


// int main(){
//     myfloat a(POSITIVE,1, 1, 3, 3); //0.5
//     myfloat b(POSITIVE,1, 1, 3, 3); //0.5
//     myfloat c(POSITIVE,0, 0, 3, 3); //0.0
//     a.setValue(POSITIVE, 1, 1);
//     b.setValue(POSITIVE, 1, 2); // 1.0
//     c.setValue(0.125); // 0.125

//     std::cout << a << std::endl;
//     std::cout << b << std::endl;
//     std::cout << c << std::endl;
    
//     myfloat add, sub, mul, div;
    
//     add = a + b; // 1.5
//     sub = a - b; // -0.5
//     mul = a * b; // 0.5
//     div = a / a; // 1.0
    
//     std::cout << add << std::endl;
//     std::cout << sub << std::endl;
//     std::cout << mul << std::endl;
//     std::cout << div << std::endl;


//     return 0;
// }