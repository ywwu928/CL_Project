#include<iostream>
using namespace std;

class MyDataTyoe
{
private:
    unsigned int s : 1; // sign
    unsigned int e : 8; // exponent
    unsigned int m : 23; // mantissa
public:
    // default constructor
    MyDataTyoe() : s(0), e(0), m(0) {}

    // copy constructor
    MyDataTyoe(const MyDataTyoe& object) {
        s = object.s;
        e = object.e;
        m = object.m;
    }

    // contructor
    MyDataTyoe(float fp32) {
        // TODO
        // can convert fp32 to MyDataTyoe here
    }

    /* operand overload */
    MyDataTyoe operator + (const MyDataTyoe& object) {
        // TODO
        MyDataTyoe result;
        return result;
    }

    MyDataTyoe operator * (const MyDataTyoe& object) {
        // TODO
        MyDataTyoe result;
        return result;
    }

    void print() {
        cout << "sign: " << s << ", exponent: " << e << ", mantissa: " << m << endl;
    }
};