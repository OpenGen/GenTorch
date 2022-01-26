#ifndef NORMAL_H
#define NORMAL_H

class Normal {
public:
    double logpdf(double value) const;
    double simulate() const;
};

#endif
