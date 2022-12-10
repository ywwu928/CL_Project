#include <iostream>

// gives an integer type fitting in N bytes
// template <int N>
// struct base_int_t
// {
//     typedef int type;
// };

// template <>
// struct base_int_t<1>
// {
//     typedef unsigned char type;
// };

// template <>
// struct base_int_t<2>
// {
//     typedef unsigned short type;
// };

// template <int EXP_BITS, int MANTISSA_BITS>
// struct fp_float
// {
//     // template argument is the number of bytes required
//     typedef typename base_int_t<(EXP_BITS + MANTISSA_BITS + 7) / 8>::type type;
//     type mantissa : MANTISSA_BITS;
//     type exponent : EXP_BITS;
// };

enum SIGN
{
    POSITIVE,
    NEGATIVE
};

class myfloat
{
private:
    int mantissa;
    int mantissa_mask;
    int mantissa_bits;
    int exponent;
    int exponent_mask;
    int exponent_bits;
    SIGN sign;
    double value; // this is the actual value of the float
    bool iszero;
public:
    myfloat(SIGN sign, int mantissa, int exponent, int mantissa_bits, int exponent_bits){
        this->mantissa_mask = (1 << mantissa_bits) - 1;
        this->exponent_mask = (1 << exponent_bits) - 1;
        this->mantissa = mantissa;
        this->exponent = exponent;
        this->sign = sign;
        this->mantissa_bits = mantissa_bits;
        this->exponent_bits = exponent_bits;
        this->setValue(sign, mantissa, exponent);
    }
    myfloat(myfloat& m){
        this->mantissa = m.getMantissa();
        this->exponent = m.getExponent();
        this->mantissa_mask = m.getMantissaMask();
        this->exponent_mask = m.getExponentMask();
        this->mantissa_bits = m.getMantissaBits();
        this->exponent_bits = m.getExponentBits();
        this->sign = m.getSign();
        this->value = m.getValue();
        this->iszero = m.getIsZero();
    }
    myfloat(){
        this->mantissa = 0;
        this->exponent = 0;
        this->mantissa_mask = 0;
        this->exponent_mask = 0;
        this->mantissa_bits = 0;
        this->exponent_bits = 0;
        this->sign = POSITIVE;
        this->value = 0;
        this->iszero = true;
    }
    ~myfloat(){};
    void setValue(float num);
    void setValue(SIGN sign, int mantissa, int exponent);
    void setMantissa(int mantissa){ 
        this->mantissa = mantissa & this->mantissa_mask;}
    void setExponent(int exponent){
        this->exponent = exponent & this->exponent_mask;}
    void setSign(SIGN sign){this->sign = sign;}
    void setMantissaMask(int mantissa_mask){this->mantissa_mask = mantissa_mask;}
    void setExponentMask(int exponent_mask){this->exponent_mask = exponent_mask;}
    void setMantissaBits(int mantissa_bits){this->mantissa_bits = mantissa_bits;}
    void setExponentBits(int exponent_bits){this->exponent_bits = exponent_bits;}
    void setIsZero(bool iszero){this->iszero = iszero;}

    double getValue(){ return this->value;}
    int getMantissa(){ return this->mantissa;}
    int getExponent(){ return this->exponent;}
    int getMantissaBits(){ return this->mantissa_bits;}
    int getExponentBits(){ return this->exponent_bits;}
    int getMantissaMask(){ return this->mantissa_mask;}
    int getExponentMask(){ return this->exponent_mask;}
    bool getIsZero(){ return this->iszero;}
    bool isZero(){ return this->iszero;}

    SIGN getSign(){ return this->sign;}

    void print();

    void normalize();
    //overload operators
    // myfloat operator+(myfloat &rhs);
    // myfloat operator-(myfloat &rhs);
    // myfloat operator*(myfloat &rhs);
    // myfloat operator/(myfloat &rhs);
    // myfloat operator=(myfloat &rhs);
    // myfloat operator==(myfloat &rhs);
    // myfloat operator>(myfloat &rhs);
    // myfloat operator>=(myfloat &rhs);
    // myfloat operator<(myfloat &rhs);
    // myfloat operator<=(myfloat &rhs);
    

    friend myfloat operator+(myfloat &lhs, myfloat &rhs);
    friend myfloat operator-(myfloat &lhs, myfloat &rhs);
    friend myfloat operator*(myfloat &lhs, myfloat &rhs);
    friend myfloat operator/(myfloat &lhs, myfloat &rhs);
    friend bool operator==(myfloat &lhs, myfloat &rhs);
    friend bool operator>(myfloat &lhs, myfloat &rhs);
    friend bool operator>=(myfloat &lhs, myfloat &rhs);
    friend bool operator<(myfloat &lhs, myfloat &rhs);
    friend bool operator<=(myfloat &lhs, myfloat &rhs);
    friend std::ostream& operator<<(std::ostream& os, const myfloat& mf);

};

void print(myfloat a){
    std::cout << a.getValue() << std::endl;
}

myfloat double2myfloat(double num){
    myfloat result;
    //get bit representation of num
    long long *p = (long long*)&num;
    long long bit_rep = *p;
    long long exponent = (bit_rep >> 52) & 0x7FF;
    long long mantissa = bit_rep & 0xFFFFFFFFFFFFF;

    result.setExponent(exponent);
    result.setMantissa(mantissa);
    return result;
}