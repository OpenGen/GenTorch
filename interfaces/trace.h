#ifndef GEN_TRACE_H
#define GEN_TRACE_H

#include <any>

#include "interfaces/trie.h"

class Trace {
public:
    [[nodiscard]] virtual Trie get_choice_trie() const = 0;
    [[nodiscard]] virtual double get_score() const = 0;
    [[nodiscard]] virtual std::any get_return_value() const = 0;
};


#endif //GEN_TRACE_H
