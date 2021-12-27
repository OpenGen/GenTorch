#pragma once

#include <vector>
#include <torch/torch.h>

using torch::Tensor;
using std::vector;
using std::pair;

// Data types that can be used as arguments and return values of generative functions

// ************
// ** Tensor **
// ************

vector<Tensor> unroll(const Tensor& args);
Tensor roll(const vector<Tensor>& unrolled, const Tensor& value);

// *************************
// ** pair<Tensor,Tensor> **
// *************************

vector<Tensor> unroll(const pair<Tensor, Tensor>& args);
pair<Tensor, Tensor> roll(const vector<Tensor>& unrolled, const pair<Tensor, Tensor>& value);

// **********************
// ** pair<Tensor,int> **
// **********************

vector<Tensor> unroll(const pair<Tensor, int>& args);
pair<Tensor, int> roll(const vector<Tensor>& unrolled, const pair<Tensor, int>& value);

// ********************
// ** vector<Tensor> **
// ********************

vector<Tensor> unroll(const vector<Tensor>& args);
vector<Tensor> roll(const vector<Tensor>& unrolled, const vector<Tensor>& value);
