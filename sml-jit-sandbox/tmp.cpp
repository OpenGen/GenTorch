#include "Interface.h"
#include "Normal.h"
std::pair<double,double> importance(const ChoiceDict& constraints) {
    double log_weight_ = 0.0;
    auto a = Normal().simulate();
    auto b = constraints.get_value(2);
    log_weight_ += Normal().logpdf(b);
    auto c = Normal().simulate();
    return {c, log_weight_};
}
