#include "model.h"

#include <utility>
#include <tuple>
#include <vector>
#include <stdexcept>
#include <array>

#include <gen/utils/randutils.h>


// Model and trace implementations



void Trace::revert() {
    throw std::logic_error("revert not implemented");
}

std::unique_ptr<Trace> Trace::fork() {
    return std::unique_ptr<Trace>(new Trace(model_, *parameters_, states_));
}
