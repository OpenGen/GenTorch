#include <gen/conversions.h>

#include <torch/torch.h>

#include <vector>

using torch::Tensor;
using std::vector;
using std::pair;

namespace gen {

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