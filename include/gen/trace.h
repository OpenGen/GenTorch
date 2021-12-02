#ifndef GEN_TRACE_H
#define GEN_TRACE_H

#include <any>

#include "trie.h"

class Trace {
public:
    /**
     * @return the choice trie.
     */
    [[nodiscard]] virtual Trie get_choice_trie() const = 0;

    /**
     * @return the log joint density.
     */
    [[nodiscard]] virtual double get_score() const = 0;

    /**
     * @return the return value of the generative function.
     */
    [[nodiscard]] virtual std::any get_return_value() const = 0;
};


#endif //GEN_TRACE_H
