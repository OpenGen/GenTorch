#include <gen/conversions.h>

#include <torch/torch.h>

#include <vector>
#include <string>
#include <sstream>

using torch::Tensor;
using std::vector;
using std::pair;

namespace gen {

    std::string __attribute__ ((noinline)) print(const Tensor& tensor) {
        std::stringstream ss;
        ss << tensor;
        std::cout << tensor << std::flush;
        return ss.str();
    }

// *************
// ** Nothing **
// *************

    Nothing zero_gradient(const Nothing& value) {
        return nothing;
    }

// *************************************
// ** Tensor (scalar numerical value) **
// *************************************

pair<size_t, Tensor> roll(const vector<Tensor> &unrolled, size_t start, const Tensor &value) {
    return {start + 1, unrolled[start]};
}

void unroll(vector<Tensor> &unrolled, const Tensor &value) {
    unrolled.emplace_back(value);
}

}