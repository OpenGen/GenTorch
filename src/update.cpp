#include <unordered_map>
#include <vector>
#include <utility>
#include <iostream>
#include <memory>
#include <cassert>
#include <exception>

using std::cout, std::endl;

struct ForgetException : public std::exception { };

struct Subtrace {
    std::string value;
    std::string value_alternate;
    bool can_be_reverted;
    explicit Subtrace(std::string value_) : value(value_), value_alternate(""), can_be_reverted(false) {}
    Subtrace(const Subtrace& other) = default;
    void update(std::string new_value, bool save, bool forget=false) {
        if (save) {
            std::swap(value, value_alternate);
            value = new_value;
            can_be_reverted = true;
        } else {
            value = new_value;
        }
        if (forget) {
            value += "_forget";
        }
    }
    void revert() {
        if (can_be_reverted) {
            std::swap(value, value_alternate);
        } else {
            throw std::logic_error("cannot be reverted");
        }
    }
};

struct SubtraceRecord {
    bool was_active;
    bool is_active;
    bool save;
    bool revert_on_restore;
    std::unique_ptr<Subtrace> subtrace;
    SubtraceRecord(bool active_, bool save_, std::unique_ptr<Subtrace>&& subtrace_) :
            was_active{false}, is_active{active_}, save{save_}, revert_on_restore{false}, subtrace{std::move(subtrace_)} {}
    SubtraceRecord(SubtraceRecord&& other) noexcept = default;
};

class Trace {
    std::unordered_map<size_t,SubtraceRecord> subtraces_{};
    bool can_be_reverted_{false};
    size_t num_active_subtraces_{0};
public:
    Trace() = default;

    bool has_subtrace(size_t address) const {
        auto it = subtraces_.find(address);
        return it != subtraces_.end();
    }

    bool has_active_subtrace(size_t address) const {
        auto it = subtraces_.find(address);
        if (it != subtraces_.end()) {
            auto& record = it->second;
            return record.is_active;
        } else {
            return false;
        }
    }

    const Subtrace& subtrace(size_t address) const {
        auto it = subtraces_.find(address);
        auto& record = it->second;
        return *record.subtrace.get();
    }

    size_t num_subtraces() const {
        return num_active_subtraces_;
    }

    void update(std::vector<std::pair<size_t, std::string>> visited, bool save) {

        // mark all existing records
        for (auto& [address, record] : subtraces_) {
            // if save, then all inactive records will be un-saved and all active records will be saved
            // otherwise, we do not change the save status of records
            if (save)
                record.save = record.is_active;
            record.was_active = record.is_active;
            record.is_active = false;
        }

        // visit records
        num_active_subtraces_ = 0;
        for (auto& [address, new_value] : visited) {
            num_active_subtraces_ += 1;
            auto it = subtraces_.find(address);
            if (it != subtraces_.end()) {
                // previous record found
                SubtraceRecord& record = it->second;
                if (record.was_active) {
                    // do a regular update call
                    record.subtrace->update(new_value, save);
                } else {
                    // do an in-place generate update call
                    record.subtrace->update(new_value, false, true);
                }
                record.is_active = true;
            } else {
                // no previous record found
                std::unique_ptr<Subtrace> subtrace = std::make_unique<Subtrace>(new_value);
                bool is_active_ = true;
                bool save_ = false;
                SubtraceRecord record{is_active_, save_, std::move(subtrace)};
                subtraces_.emplace(address, std::move(record));
            }

        }

        for (auto it = subtraces_.begin(); it != subtraces_.end();) {
            auto& [address, record] = *it;

            if (!record.is_active && !record.save) {

                // destruct records that are inactive and not marked as saved
                it = subtraces_.erase(it);
            } else {

                // records that were already inactive and saved, and are not re-saved, should be reverted on restore
                if (!save && record.was_active && !record.is_active) {
                    assert(record.save);
                    record.revert_on_restore = true;
                }

                it++;
            }
        }


        can_be_reverted_ = can_be_reverted_ || save;
    }

    void revert() {
        if (!can_be_reverted_)
            throw std::logic_error("cannot be reverted");
        can_be_reverted_ = false;

        num_active_subtraces_ = 0;
        for (auto it = subtraces_.begin(); it != subtraces_.end();) {
            auto& [address, record] = *it;

            if (record.save) {
                // if it was already active it should be reverted
                // if it was not already active, then it may need to be reverted, depending on revert_on_restore
                if (record.is_active || record.revert_on_restore)
                    record.subtrace->revert();

                // anything that is marked as save should be marked as active
                record.is_active = true;
                num_active_subtraces_ += 1;
            }

            if (!record.save) {

                // anything that is not marked as saved should be destructed
                it = subtraces_.erase(it);
            } else {

                // anything that was marked as save will be unmarked
                record.save = false;

                // reset
                record.revert_on_restore = false;

                // continue
                it++;
            }

        }
    }
};

int main() {

    Trace trace;

    // initialize
    trace.update({{1, "a"}, {2, "b"}}, false);
    assert(trace.num_subtraces() == 2);
    assert(trace.has_active_subtrace(1));
    assert(trace.has_active_subtrace(2));
    assert(trace.subtrace(1).value == "a");
    assert(trace.subtrace(2).value == "b");

    // update (save = F)
    trace.update({{3, "c"}, {2, "b"}}, false);
    assert(trace.num_subtraces() == 2);
    assert(!trace.has_active_subtrace(1));
    assert(trace.has_active_subtrace(2));
    assert(trace.has_active_subtrace(3));
    assert(trace.subtrace(2).value == "b");
    assert(trace.subtrace(3).value == "c");

    // update (save = F)
    trace.update({{4, "d"}, {2, "e"}}, false);
    assert(trace.num_subtraces() == 2);
    assert(!trace.has_subtrace(1));
    assert(trace.has_active_subtrace(2));
    assert(!trace.has_subtrace(3));
    assert(trace.has_active_subtrace(4));
    assert(trace.subtrace(2).value == "e");
    assert(trace.subtrace(4).value == "d");

    // update (save = T)
    // add 5 => "g", update value of 4 to "f", and remove 2 (but save it)
    trace.update({{5, "g"}, {4, "f"}}, true);
    assert(trace.num_subtraces() == 2);
    assert(!trace.has_active_subtrace(1));
    assert(trace.has_subtrace(2) && !trace.has_active_subtrace(2)); // saved
    assert(!trace.has_active_subtrace(3));
    assert(trace.has_active_subtrace(4) && trace.subtrace(4).value == "f");
    assert(trace.has_active_subtrace(5) && trace.subtrace(5).value == "g");

    // update (save = T)
    // add 6 => "i", update value of 5 to "h", and remove 4 (but save it)
    trace.update({{6, "i"}, {5, "h"}}, true);
    assert(trace.num_subtraces() == 2);
    assert(!trace.has_subtrace(1));
    assert(!trace.has_subtrace(2));
    assert(!trace.has_subtrace(3));
    assert(trace.has_subtrace(4) && !trace.has_active_subtrace(4)); // saved, revert_on_restore should be false here
    assert(trace.has_active_subtrace(5) && trace.subtrace(5).value == "h");
    assert(trace.has_active_subtrace(6) && trace.subtrace(6).value == "i");

    // revert to 5 => "g" and 4 => "f"
    trace.revert();
    assert(trace.num_subtraces() == 2);
    assert(!trace.has_subtrace(1));
    assert(!trace.has_subtrace(2));
    assert(!trace.has_subtrace(3));
    assert(trace.has_active_subtrace(4));
    assert(trace.has_active_subtrace(5));
    assert(!trace.has_subtrace(6));
    assert(trace.subtrace(4).value == "f");
    assert(trace.subtrace(5).value == "g");

    // attempt to revert twice
    bool threw = false;
    try {
        trace.revert();
    } catch (const std::logic_error&) {
        threw = true;
    }
    assert(threw);

    // update (save = T)
    // add 6 => "k", update value of 5 to "j", and remove 4 (but save it)
    trace.update({{6, "k"}, {5, "j"}}, true);
    assert(trace.num_subtraces() == 2);
    assert(!trace.has_subtrace(1));
    assert(!trace.has_subtrace(2));
    assert(!trace.has_subtrace(3));
    assert(trace.has_subtrace(4) && !trace.has_active_subtrace(4));
    assert(trace.has_active_subtrace(5) && trace.subtrace(5).value == "j");
    assert(trace.has_active_subtrace(6) && trace.subtrace(6).value == "k");

    // update (save = F)
    // add 7 => "m", update value of 6 to "l", and remove 5
    trace.update({{7, "m"}, {6, "l"}}, false);
    assert(trace.num_subtraces() == 2);
    assert(!trace.has_subtrace(1));
    assert(!trace.has_subtrace(2));
    assert(!trace.has_subtrace(3));
    assert(trace.has_subtrace(4) && !trace.has_active_subtrace(4)); // saved, revert_on_restore should not be set TODO
    assert(trace.has_subtrace(5) && !trace.has_active_subtrace(5)); // saved
    assert(trace.has_active_subtrace(6));
    assert(trace.has_active_subtrace(7));
    assert(trace.subtrace(6).value == "l");
    assert(trace.subtrace(7).value == "m");

    // update (save = F)
    // add 8 => "o", update value of 7 to "n", and remove 6
    trace.update({{8, "o"}, {7, "n"}}, false);
    assert(trace.num_subtraces() == 2);
    assert(!trace.has_subtrace(1));
    assert(!trace.has_subtrace(2));
    assert(!trace.has_subtrace(3));
    assert(trace.has_subtrace(4) && !trace.has_active_subtrace(4)); // saved, revert_on_restore should not be set TODO
    assert(trace.has_subtrace(5) && !trace.has_active_subtrace(5)); // saved
    assert(!trace.has_subtrace(6));
    assert(trace.has_active_subtrace(7));
    assert(trace.has_active_subtrace(8));
    assert(trace.subtrace(7).value == "n");
    assert(trace.subtrace(8).value == "o");

    // revert to 5 => "g" and 4 => "f"
    trace.revert();
    assert(trace.num_subtraces() == 2);
    assert(!trace.has_subtrace(1));
    assert(!trace.has_subtrace(2));
    assert(!trace.has_subtrace(3));
    assert(trace.has_active_subtrace(4)); // saved
    assert(trace.has_active_subtrace(5)); // saved
    assert(!trace.has_subtrace(6));
    assert(!trace.has_subtrace(7));
    assert(!trace.has_subtrace(8));
    assert(trace.subtrace(4).value == "f");
    assert(trace.subtrace(5).value == "g");

    // update (save = T)
    // add 6 => "k", update value of 5 to "j", and remove 4 (but save it)
    trace.update({{6, "k"}, {5, "j"}}, true);
    assert(trace.num_subtraces() == 2);
    assert(!trace.has_subtrace(1));
    assert(!trace.has_subtrace(2));
    assert(!trace.has_subtrace(3));
    assert(trace.has_subtrace(4) && !trace.has_active_subtrace(4)); // saved
    assert(trace.has_active_subtrace(5));
    assert(trace.has_active_subtrace(6));
    assert(!trace.has_subtrace(7));
    assert(!trace.has_subtrace(8));
    assert(trace.subtrace(5).value == "j");
    assert(trace.subtrace(6).value == "k");

    // update (save = T)
    // re-introduce 4
    trace.update({{4, "p"}, {6, "k"}, {5, "j"}}, true);
    assert(trace.num_subtraces() == 3);
    assert(!trace.has_subtrace(1));
    assert(!trace.has_subtrace(2));
    assert(!trace.has_subtrace(3));
    assert(trace.has_active_subtrace(4));
    assert(trace.has_active_subtrace(5));
    assert(trace.has_active_subtrace(6));
    assert(!trace.has_subtrace(7));
    assert(!trace.has_subtrace(8));
    assert(trace.subtrace(4).value == "p_forget");
    assert(trace.subtrace(5).value == "j");
    assert(trace.subtrace(6).value == "k");

    // update (save = F)
    // remove everything
    trace.update({}, false);
    assert(trace.num_subtraces() == 0);
    assert(!trace.has_subtrace(1));
    assert(!trace.has_subtrace(2));
    assert(!trace.has_subtrace(3));
    assert(!trace.has_subtrace(4));
    assert(trace.has_subtrace(5) && !trace.has_active_subtrace(5)); // saved
    assert(trace.has_subtrace(6) && !trace.has_active_subtrace(6)); // saved
    assert(!trace.has_subtrace(7));
    assert(!trace.has_subtrace(8));

    // update (save = F)
    // re-introduce 5
    trace.update({{5, "q"}}, false);
    assert(trace.num_subtraces() == 1);
    assert(!trace.has_subtrace(1));
    assert(!trace.has_subtrace(2));
    assert(!trace.has_subtrace(3));
    assert(!trace.has_subtrace(4));
    assert(trace.has_active_subtrace(5));
    assert(trace.has_subtrace(6) && !trace.has_active_subtrace(6)); // saved
    assert(!trace.has_subtrace(7));
    assert(!trace.has_subtrace(8));
    assert(trace.subtrace(5).value == "q_forget");

    // revert to {6, "k"}, {5, "j"}
    trace.revert();
    assert(trace.num_subtraces() == 2);
    assert(!trace.has_subtrace(1));
    assert(!trace.has_subtrace(2));
    assert(!trace.has_subtrace(3));
    assert(!trace.has_subtrace(4));
    assert(trace.has_active_subtrace(5));
    assert(trace.has_active_subtrace(6));
    assert(!trace.has_subtrace(7));
    assert(!trace.has_subtrace(8));
    assert(trace.subtrace(5).value == "j");
    assert(trace.subtrace(6).value == "k");

}