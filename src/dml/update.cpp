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

// transitions
// update (U)
// update save (US)
// update that restores an inactive subtrace (UR)
// update save that restores an inactive subtrace (URS)
// update that removes an active subtrace (UM)
// update save that removes an inactive subtrace (UMS)
// revert (R)

// ACTIVE ----- U -----> ACTIVE                             [call (has_subtrace)]
// ACTIVE ----- US ----> ACTIVE_SAVED                       [call (has_subtrace)]
// ACTIVE ----- UM ----> destruct                           [update_post_visit]
// ACTIVE ----- UMS ---> INACTIVE_SAVED                     [update_post_visit]            xxx
// ACTIVE ------ R ----> destruct                           [revert_visit]
// ACTIVE_SAVED ----- U ----> ACTIVE_SAVED                  [call (has_subtrace)]
// ACTIVE_SAVED ----- US ---> ACTIVE_SAVED                  [call (has_subtrace)]
// ACTIVE_SAVED ----- UM ---> INACTIVE_SAVED_REVERT         [update_post_visit]            xxx
// ACTIVE_SAVED ----- UMS --> INACTIVE_SAVED                [update_post_visit]            xxx
// ACTIVE_SAVED ----- R ----> ACTIVE (+ revert)             [revert_visit]
// INACTIVE_SAVED --- U ----> INACTIVE_SAVED                [update_post_visit]
// INACTIVE_SAVED --- US ---> destruct                      [update_post_visit]
// INACTIVE_SAVED --- UR ---> ACTIVE_SAVED                  [call (has_subtrace)]
// INACTIVE_SAVED --- URS --> ACTIVE                        [call (has_subtrace)]
// INACTIVE_SAVED --- R ----> ACTIVE                        [revert_visit]
// INACTIVE_SAVED_REVERT --- U ----> INACTIVE_SAVED_REVERT  [update_post_visit]
// INACTIVE_SAVED_REVERT --- US ---> destruct               [update_post_visit]
// INACTIVE_SAVED_REVERT --- UR ---> ACTIVE_SAVED           [call (has_subtrace)]
// INACTIVE_SAVED_REVERT --- URS --> ACTIVE                 [call (has_subtrace)]
// INACTIVE_SAVED_REVERT --- R ----> ACTIVE (+ revert)      [revert_visit]



void update_pre_visit(Trie<SubtraceRecord>& trie) {
    // recursively walk the trie and mark all existing records
    for (auto& [key, subtrie] : trie.subtries()) {
        if (subtrie.has_value()) {
            auto& record = subtrie.get_value();
            record.was_active = (record.state == RecordState::Active || record.state == RecordState::ActiveSaved);
            record.is_active = false;
        } else {
            assert(!subtrie.empty());
            update_pre_visit(subtrie);
        }
    }
}

[[nodiscard]] bool has_subtrace(const Trie<SubtraceRecord>& trie, const Address& address) {
    return trie.has_subtrie(address) && trie.get_subtrie(address).has_value();
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
            auto &record = subtrie.get_value();
            bool removed = record.was_active && !record.is_active;
            if (removed) {
                const ChoiceTrie &backward_constraints_subtrie = record.subtrace->choices();
                if (!backward_constraints_subtrie.empty())
                    backward_constraints.set_subtrie(address, backward_constraints_subtrie);
                unvisited_score += record.subtrace->score();
                if (record.state == RecordState::Active && !save) {
                    // ACTIVE ----- UM ----> destruct
                    it = subtries_map.erase(it);
                } else if (record.state == RecordState::Active && save) {
                    // ACTIVE ----- UMS ---> INACTIVE_SAVED
                    record.state = RecordState::InactiveSaved;
                    it++;
                } else if (record.state == RecordState::ActiveSaved && !save) {
                    // ACTIVE_SAVED ----- UM ---> INACTIVE_SAVED_REVERT
                    record.state = RecordState::InactiveSavedRevert;
                    it++;
                } else if (record.state == RecordState::ActiveSaved && save) {
                    // ACTIVE_SAVED ----- UMS --> INACTIVE_SAVED
                    record.state = RecordState::InactiveSaved;
                    it++;
                } else {
                    assert(false);
                }
            } else if (!record.is_active) {
                if (!save) {
                    // INACTIVE_SAVED --- U ----> INACTIVE_SAVED
                    // INACTIVE_SAVED_REVERT --- U ----> INACTIVE_SAVED_REVERT
                    it++;
                } else {
                    // INACTIVE_SAVED --- US ---> destruct
                    // INACTIVE_SAVED_REVERT --- US ---> destruct
                    it = subtries_map.erase(it);
                }
            } else {
                it++;
            }
        } else {
            ChoiceTrie backward_constraints_subtrie;
            unvisited_score += update_post_visit(subtrie, save, backward_constraints_subtrie);
            if (!backward_constraints_subtrie.empty())
                backward_constraints.set_subtrie(address, backward_constraints_subtrie);
            it++;
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
            if (record.state == RecordState::Active) {
                // ACTIVE ------ R ----> destruct
                it = subtries.erase(it);
            } else if (record.state == RecordState::ActiveSaved) {
                // ACTIVE_SAVED ----- R ----> ACTIVE (+ revert)
                record.state = RecordState::Active;
                record.subtrace->revert();
                it++;
            } else if (record.state == RecordState::InactiveSaved) {
                // INACTIVE_SAVED --- R ----> ACTIVE
                record.state = RecordState::Active;
                it++;
            } else {
                // INACTIVE_SAVED_REVERT --- R ----> ACTIVE (+ revert)
                assert(record.state == RecordState::InactiveSavedRevert);
                record.state = RecordState::Active;
                record.subtrace->revert();
                it++;
            }
        } else {
            assert(!subtrie.empty());
            revert_visit(subtrie);
            it++;
        }
    }
}



}