#include<iostream>
using namespace std;

class MyDataType
{
private:
    unsigned int s : 1; // sign
    unsigned int e : 8; // exponent
    unsigned int m : 23; // mantissa
public:
    // default constructor
    MyDataType () : s(0), e(0), m(0) {}

    // copy constructor
    MyDataType (MyDataType& object) {
        s = object.s;
        e = object.e;
        m = object.m;
    }

    // contructor
    MyDataType (float fp32) {
        // TODO
        // can convert fp32 to MyDataType here
    }

    // destructor
    ~MyDataType () {}

    /* operand overload */
    MyDataType operator + (const MyDataType& object) {
        // TODO
        MyDataType result;
        return result;
    }

    MyDataType operator * (const MyDataType& object) {
        // TODO
        MyDataType result;
        return result;
    }

    bool operator > (const MyDataType& object) {
        // TODO
        bool result;
        return result;
    }

    bool operator < (const MyDataType& object) {
        // TODO
        bool result;
        return result;
    }

    void print() {
        cout << "sign: " << s << ", exponent: " << e << ", mantissa: " << m << endl;
    }
};