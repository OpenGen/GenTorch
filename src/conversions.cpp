#include <gen/conversions.h>

#include <torch/torch.h>

#include <vector>

using torch::Tensor;
using std::vector;
using std::pair;

// *************************************
// ** Tensor (scalar numerical value) **
// *************************************

pair<size_t,Tensor> roll(const vector<Tensor>& unrolled, size_t start, const Tensor& value) {
    return {start + 1, unrolled[start]};
}

void unroll(vector<Tensor>& unrolled, const Tensor& value) {
    unrolled.emplace_back(value);
}
//
//// *************************
//// ** pair<Tensor,Tensor> **
//// *************************
//
//vector<Tensor> unroll(const pair<Tensor, Tensor>& args) {
//    vector<Tensor> tensors {args.first, args.second};
//    return move(tensors);
//}
//
//pair<Tensor, Tensor> roll(const vector<Tensor>& unrolled, const pair<Tensor, Tensor>& value) {
//    assert(unrolled.size() == 2);
//    return {unrolled[0], unrolled[1]};
//}
//
//// **********************
//// ** pair<Tensor,int> **
//// **********************
//
//vector<Tensor> unroll(const pair<Tensor, int>& args) {
//    vector<Tensor> tensors {args.first};
//    return move(tensors);
//}
//
//pair<Tensor, int> roll(const vector<Tensor>& unrolled, const pair<Tensor, int>& value) {
//    assert(unrolled.size() == 1);
//    return {unrolled[0], value.second};
//}

//// ********************
//// ** vector<Tensor> **
//// ********************
//
//vector<Tensor> unroll(const vector<Tensor>& args) {
//    return args;
//}
//
//vector<Tensor> roll(const vector<Tensor>& unrolled, const vector<Tensor>& value) {
//    return unrolled;
//}