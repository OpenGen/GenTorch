/* Copyright 2021-2022 Massachusetts Institute of Technology

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
        limitations under the License.
==============================================================================*/

#include <catch2/catch.hpp>
#include <gentorch/trie.h>
#include <gentorch/dml/dml.h>
#include <gentorch/distributions/bernoulli.h>
#include <gentorch/conversions.h>

#include <sstream>

#include <torch/torch.h>

using torch::Tensor;
using torch::TensorOptions;
using torch::tensor;

using gentorch::Nothing, gentorch::nothing;
using gentorch::dml::DMLGenFn;
using gentorch::distributions::bernoulli::Bernoulli;
using gentorch::ChoiceTrie;
using gentorch::EmptyModule;

using gentl::SimulateOptions;
using gentl::GenerateOptions;
using gentl::UpdateOptions;

Address addr(int i) {
    std::stringstream ss;
    ss << i << endl;
    return Address{ss.str()};
}

class Foo : public DMLGenFn<Foo, std::pair<int,int>, Nothing, EmptyModule> {
public:
    explicit Foo(int addr1, int addr2) : DMLGenFn<M,A,R,P>({addr1, addr2}) {}
    Foo(const Foo& other) : DMLGenFn(other) {}
    template <typename Tracer>
    return_type forward(Tracer& tracer) const {
        auto [addr1, addr2] = tracer.get_args();
        tracer.call(addr(addr1), Bernoulli(tensor(0.5)));
        tracer.call(addr(addr2), Bernoulli(tensor(0.5)));
        return nothing;
    }
};

TEST_CASE("update long", "[dml]") {
    c10::InferenceMode guard{true};
    std::random_device rd{};
    std::mt19937 rng{rd()};
    EmptyModule parameters;
    Foo model {1, 2};
    auto change = [](int addr1, int addr2) { return gentl::change::UnknownChange<Foo>(Foo{addr1, addr2}); };
    auto constraint = [](int addr1, int addr2, bool value1, bool value2) {
        ChoiceTrie constraints;
        constraints.set_value(addr(addr1), value1, false);
        constraints.set_value(addr(addr2), value2, false);
        return constraints;
    };
    auto has_choice = [](const ChoiceTrie& choices, int addr1, bool value1) {
        return std::any_cast<bool>(choices.get_value(addr(addr1))) == value1;
    };
    auto no_choice = [](const ChoiceTrie& choices, int addr1) {
        return !choices.has_subtrie(addr(addr1));
    };

    // initialize
    auto [trace, log_weight] = model.generate(
            rng, parameters, constraint(1, 2, false, false),
            GenerateOptions().precompute_gradient(false));

    // update 1 => true, 2 => true
    {
        trace->update(rng, change(1, 2), constraint(1, 2, true, true), UpdateOptions().save(false));
        ChoiceTrie choices = trace->choices();
        REQUIRE(has_choice(choices, 1, true));
        REQUIRE(has_choice(choices, 2, true));
        ChoiceTrie backward = trace->backward_constraints();
        REQUIRE(has_choice(backward, 1, false));
        REQUIRE(has_choice(backward, 2, false));
    }

    // update 2 => false, 3 => ftracealse
    {
        trace->update(rng, change(2, 3), constraint(2, 3, false, false), UpdateOptions().save(false));
        ChoiceTrie choices = trace->choices();
        REQUIRE(no_choice(choices, 1));
        REQUIRE(has_choice(choices, 2, false));
        REQUIRE(has_choice(choices, 3, false)); // FAILS
        ChoiceTrie backward = trace->backward_constraints();
        REQUIRE(has_choice(backward, 1, true));
        REQUIRE(has_choice(backward, 2, true));
    }

    // update 2 => true, 4 => true, and remove 3
    {
        trace->update(rng, change(2, 4), constraint(2, 4, true, true), UpdateOptions().save(true));
        ChoiceTrie choices = trace->choices();
        REQUIRE(no_choice(choices, 1));
        REQUIRE(no_choice(choices, 3));
        REQUIRE(has_choice(choices, 2, true));
        REQUIRE(has_choice(choices, 4, true));
        ChoiceTrie backward = trace->backward_constraints();
        REQUIRE(has_choice(backward, 2, false));
        REQUIRE(has_choice(backward, 3, false));
    }

    // update 4 => false, 5 => false, and remove 2 (but save it)
    {
        trace->update(rng, change(4, 5), constraint(4, 5, false, false), UpdateOptions().save(true));
        ChoiceTrie choices = trace->choices();
        REQUIRE(no_choice(choices, 1));
        REQUIRE(no_choice(choices, 2));
        REQUIRE(no_choice(choices, 3));
        REQUIRE(has_choice(choices, 4, false));
        REQUIRE(has_choice(choices, 5, false));
        ChoiceTrie backward = trace->backward_constraints();
        REQUIRE(has_choice(backward, 2, true));
        REQUIRE(has_choice(backward, 4, true));
    }

    // update 5 => true, 6 => true, and remove 4 (but save it); this deletes the inactive subtrace at 2
    {
        trace->update(rng, change(5, 6), constraint(5, 6, true, true), UpdateOptions().save(true));
        ChoiceTrie choices = trace->choices();
        REQUIRE(no_choice(choices, 1));
        REQUIRE(no_choice(choices, 2));
        REQUIRE(no_choice(choices, 3));
        REQUIRE(no_choice(choices, 4));
        REQUIRE(has_choice(choices, 5, true));
        REQUIRE(has_choice(choices, 6, true));
        ChoiceTrie backward = trace->backward_constraints();
        REQUIRE(has_choice(backward, 4, false));
        REQUIRE(has_choice(backward, 5, false));
    }

    // revert to 4 => false, 5 => false
    {
        ChoiceTrie constraints;
        trace->revert();
        ChoiceTrie choices = trace->choices();
        REQUIRE(no_choice(choices, 1));
        REQUIRE(no_choice(choices, 2));
        REQUIRE(no_choice(choices, 3));
        REQUIRE(no_choice(choices, 6));
        REQUIRE(has_choice(choices, 4, false));
        REQUIRE(has_choice(choices, 5, false));
    }

    // attempt to revert twice
    {
        bool threw = false;
        try {
            trace->revert();
        } catch (const std::logic_error &) {
            threw = true;
        }
        REQUIRE(threw);
    }

    // update to 6 => true, 5 => true, and save
    {
        ChoiceTrie constraints;
        trace->update(rng, change(5, 6), constraint(5, 6, true, true), UpdateOptions().save(true));
        ChoiceTrie choices = trace->choices();
        REQUIRE(no_choice(choices, 1));
        REQUIRE(no_choice(choices, 2));
        REQUIRE(no_choice(choices, 3));
        REQUIRE(no_choice(choices, 4));
        REQUIRE(has_choice(choices, 5, true));
        REQUIRE(has_choice(choices, 6, true));
        ChoiceTrie backward = trace->backward_constraints();
        REQUIRE(has_choice(backward, 4, false));
        REQUIRE(has_choice(backward, 5, false));
    }

    // update 6 => true, update value of 5 to false, and save
    {
        ChoiceTrie constraints;
        trace->update(rng, change(5, 6), constraint(5, 6, false, true), UpdateOptions().save(true));
        ChoiceTrie choices = trace->choices();
        REQUIRE(no_choice(choices, 1));
        REQUIRE(no_choice(choices, 2));
        REQUIRE(no_choice(choices, 3));
        REQUIRE(no_choice(choices, 4));
        REQUIRE(has_choice(choices, 5, false));
        REQUIRE(has_choice(choices, 6, true));
        ChoiceTrie backward = trace->backward_constraints();
        REQUIRE(has_choice(backward, 5, true));
        REQUIRE(has_choice(backward, 6, true));
    }

    // update (save = F)
    // add 7 => true, update value of 6 to false, and remove 5
    {
        ChoiceTrie constraints;
        trace->update(rng, change(6, 7), constraint(6, 7, false, true), UpdateOptions().save(false));
        ChoiceTrie choices = trace->choices();
        REQUIRE(no_choice(choices, 1));
        REQUIRE(no_choice(choices, 2));
        REQUIRE(no_choice(choices, 3));
        REQUIRE(no_choice(choices, 4));
        REQUIRE(no_choice(choices, 5));
        REQUIRE(has_choice(choices, 6, false));
        REQUIRE(has_choice(choices, 7, true));
        ChoiceTrie backward = trace->backward_constraints();
        REQUIRE(has_choice(backward, 5, false));
        REQUIRE(has_choice(backward, 6, true));
    }

    // update (save = F)
    // add 8 => true, update value of 7 to false, and remove 6
    {
        ChoiceTrie constraints;
        trace->update(rng, change(7, 8), constraint(7, 8, false, true), UpdateOptions().save(false));
        ChoiceTrie choices = trace->choices();
        REQUIRE(no_choice(choices, 1));
        REQUIRE(no_choice(choices, 2));
        REQUIRE(no_choice(choices, 3));
        REQUIRE(no_choice(choices, 4));
        REQUIRE(no_choice(choices, 5));
        REQUIRE(no_choice(choices, 6));
        REQUIRE(has_choice(choices, 7, false));
        REQUIRE(has_choice(choices, 8, true));
        ChoiceTrie backward = trace->backward_constraints();
        REQUIRE(has_choice(backward, 6, false));
        REQUIRE(has_choice(backward, 7, true));
    }

    // revert to 6 => true, 5 => true
    {
        trace->revert();
        ChoiceTrie choices = trace->choices();
        REQUIRE(no_choice(choices, 1));
        REQUIRE(no_choice(choices, 2));
        REQUIRE(no_choice(choices, 3));
        REQUIRE(no_choice(choices, 4));
        REQUIRE(has_choice(choices, 5, true));
        REQUIRE(has_choice(choices, 6, true));
        REQUIRE(no_choice(choices, 7));
        REQUIRE(no_choice(choices, 8));
    }

    // update (save = T)
    // add 6 => true, update value of 5 to false, and remove 4 (but save it)
    {
        ChoiceTrie constraints;
        trace->update(rng, change(5, 6), constraint(5, 6, false, true), UpdateOptions().save(true));
        ChoiceTrie choices = trace->choices();
        REQUIRE(no_choice(choices, 1));
        REQUIRE(no_choice(choices, 2));
        REQUIRE(no_choice(choices, 3));
        REQUIRE(no_choice(choices, 4));
        REQUIRE(has_choice(choices, 5, false));
        REQUIRE(has_choice(choices, 6, true));
        REQUIRE(no_choice(choices, 7));
        REQUIRE(no_choice(choices, 8));
        ChoiceTrie backward = trace->backward_constraints();
        REQUIRE(has_choice(backward, 5, true));
        REQUIRE(has_choice(backward, 6, true));
    }

    // update (save = T)
    // add reintroduce 4 and remove 5
    {
        ChoiceTrie constraints;
        trace->update(rng, change(4, 6), constraint(4, 6, false, true), UpdateOptions().save(true));
        ChoiceTrie choices = trace->choices();
        REQUIRE(no_choice(choices, 1));
        REQUIRE(no_choice(choices, 2));
        REQUIRE(no_choice(choices, 3));
        REQUIRE(has_choice(choices, 4, false)); // TODO we should check that this took the different 'forget' code path
        REQUIRE(no_choice(choices, 5));
        REQUIRE(has_choice(choices, 6, true));
        REQUIRE(no_choice(choices, 7));
        REQUIRE(no_choice(choices, 8));
        ChoiceTrie backward = trace->backward_constraints();
        REQUIRE(has_choice(backward, 5, false));
        REQUIRE(has_choice(backward, 6, true));
    }

    // update (save = F)
    // remove everything
    {
        ChoiceTrie constraints;
        trace->update(rng, change(9, 10), constraint(9, 10, true, true), UpdateOptions().save(false));
        ChoiceTrie choices = trace->choices();
        REQUIRE(no_choice(choices, 1));
        REQUIRE(no_choice(choices, 2));
        REQUIRE(no_choice(choices, 3));
        REQUIRE(no_choice(choices, 4));
        REQUIRE(no_choice(choices, 5));
        REQUIRE(no_choice(choices, 6));
        REQUIRE(no_choice(choices, 7));
        REQUIRE(no_choice(choices, 8));
        REQUIRE(has_choice(choices, 9, true));
        REQUIRE(has_choice(choices, 10, true));
        ChoiceTrie backward = trace->backward_constraints();
        REQUIRE(has_choice(backward, 4, false));
        REQUIRE(has_choice(backward, 6, true));
    }

    // update (save = F)
    // re-introduce 5
    {
        ChoiceTrie constraints;
        trace->update(rng, change(5, 10), constraint(5, 10, true, true), UpdateOptions().save(false));
        ChoiceTrie choices = trace->choices();
        REQUIRE(no_choice(choices, 1));
        REQUIRE(no_choice(choices, 2));
        REQUIRE(no_choice(choices, 3));
        REQUIRE(no_choice(choices, 4));
        REQUIRE(has_choice(choices, 5, true));
        REQUIRE(no_choice(choices, 6));
        REQUIRE(no_choice(choices, 7));
        REQUIRE(no_choice(choices, 8));
        REQUIRE(no_choice(choices, 9));
        REQUIRE(has_choice(choices, 10, true));
        ChoiceTrie backward = trace->backward_constraints();
        REQUIRE(has_choice(backward, 9, true));
        REQUIRE(has_choice(backward, 10, true));
    }

    // revert to 6 => true, 5 => false
    {
        trace->revert();
        ChoiceTrie choices = trace->choices();
        REQUIRE(no_choice(choices, 1));
        REQUIRE(no_choice(choices, 2));
        REQUIRE(no_choice(choices, 3));
        REQUIRE(no_choice(choices, 4));
        REQUIRE(has_choice(choices, 5, false));
        REQUIRE(has_choice(choices, 6, true));
        REQUIRE(no_choice(choices, 7));
        REQUIRE(no_choice(choices, 8));
        REQUIRE(no_choice(choices, 9));
        REQUIRE(no_choice(choices, 10));
    }

}