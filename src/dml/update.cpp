/* Copyright 2021 The LibGen Authors

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

#include <gen/dml/dml.h>

namespace gen::dml {


void update_pre_visit(Trie<SubtraceRecord>& trie, bool save) {
    // recursively walk the trie and mark all existing records
    for (auto& [key, subtrie] : trie.subtries()) {
        if (subtrie.has_value()) {
            auto& record = subtrie.get_value();
            // if save, then all inactive records will be un-saved and all active records will be saved
            // otherwise, we do not change the save status of records
            if (save)
                record.save = record.is_active;
            record.was_active = record.is_active;
            record.is_active = false;
        } else {
            assert(!subtrie.empty());
            update_pre_visit(subtrie, save);
        }
    }
}

[[nodiscard]] bool has_subtrace(const Trie<SubtraceRecord>& trie, const Address& address) {
    return trie.get_subtrie(address).has_value();
}

[[nodiscard]] SubtraceRecord& get_subtrace_record(const Trie<SubtraceRecord>& trie, const Address& address) {
    return trie.get_subtrie(address).get_value();
}

double update_post_visit(Trie<SubtraceRecord>& trie, bool save, ChoiceTrie& backward_constraints) {
    double unvisited_score = 0.0;
    // recursively walk the trie
    auto& subtries_map = trie.subtries();
    for (auto it = subtries_map.begin(); it != subtries_map.end();) {
        auto& [key, subtrie] = *it;
        const Address address{key};
        assert(!subtrie.empty());
        if (subtrie.has_value()) {
            auto& record = subtrie.get_value();

            // if the subtrace was_active, but is not now active, then it contributes to backward_constraints
            if (record.was_active && !record.is_active) {
                const ChoiceTrie &backward_constraints_subtrie = record.subtrace->choices();
                if (!backward_constraints_subtrie.empty())
                    backward_constraints.set_subtrie(address, backward_constraints_subtrie);
                unvisited_score += record.subtrace->score();
            }

            if (!record.is_active && !record.save) {

                // destruct records that are inactive and not marked as saved
                it = subtries_map.erase(it);
            } else {

                // records that were already inactive and saved, and are not re-saved, should be reverted on restore
                if (!save && record.was_active && !record.is_active) {
                    assert(record.save);
                    record.revert_on_restore = true;
                }
                it++;
            }
        } else {
            ChoiceTrie backward_constraints_subtrie;
            unvisited_score += update_post_visit(subtrie, save, backward_constraints_subtrie);
            if (!backward_constraints_subtrie.empty())
                backward_constraints.set_subtrie(address, backward_constraints_subtrie);
        }
    }
    return unvisited_score;
}

void revert_visit(Trie<SubtraceRecord>& trie) {
    auto& subtries = trie.subtries();
    for (auto it = subtries.begin(); it != subtries.end();) {
        auto& [key, subtrie] = *it;
        const Address address{key};
        if (subtrie.has_value()) {
            auto& record = subtrie.get_value();
            if (record.save) {
                // if it was already active it should be reverted
                // if it was not already active, then it may need to be reverted, depending on revert_on_restore
                if (record.is_active || record.revert_on_restore)
                    record.subtrace->revert();

                // anything that is marked as save should be marked as active
                record.is_active = true;
            }

            if (!record.save) {

                // anything that is not marked as saved should be destructed
                it = subtries.erase(it);
            } else {

                // anything that was marked as save will be unmarked
                record.save = false;

                // reset
                record.revert_on_restore = false;

                // continue
                it++;
            }
        } else {
            assert(!subtrie.empty());
            revert_visit(subtrie);
        }
    }
}



}