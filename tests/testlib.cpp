#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <gen/trie.h>

#include <torch/torch.h>

using torch::tensor;
using torch::Tensor;

TEST_CASE( "Quick check", "[main]" ) {

Tensor t1 = tensor(1.123);
Tensor t2 = tensor(2.234);

Trie trie { };
REQUIRE(!trie.has_value());

}