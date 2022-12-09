// Adam reference: https://arxiv.org/pdf/1412.6980.pdf
// maybe: https://arxiv.org/pdf/2004.14180.pdf

class AdamOpt 
{
  public:
    const double alpha;
    const double beta1;
    const double beta2;
    const double epsilon;

    AdamOpt(double a = 0.001, double b1 = 0.9, double b2 = 0.999, double e = 1e-8)
        : alpha(a),
          beta1(b1),
          beta2(b2),
          epsilon(e),
          
          m(0.0),
          v(0.0),
          beta1t(b1),
          beta2t(b2),
          beta1_inv(1-b1),
          beta2_inv(1-b2) {}

    ~AdamOpt();

    template <class T>
    T update(T grad);
    
  private:
    double m;   // biased 1st moment estimate
    double v;   // biased 2nd raw moment estimate

    // to reduce computations
    double beta1t;  // beta1^t
    double beta2t;  // beta2^t
    double beta1_inv;
    double beta2_inv;
};
