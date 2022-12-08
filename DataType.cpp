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
    MyDataTyoe(const MyDataTyoe& copy) {
        s = copy.s;
        e = copy.e;
        m = copy.m;
    }

    // contructor
    MyDataTyoe(float fp32) {
        // TODO
        // can convert fp32 to MyDataTyoe here
    }
};