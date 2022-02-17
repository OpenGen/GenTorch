set(DOXYGEN_EXTRACT_ALL YES)
set(DOXYGEN_BUILTIN_STL_SUPPORT YES)

doxygen_add_docs(docs
        gentorch/parameters.h
        gentorch/address.h
        gentorch/conversions.h
        gentorch/trie.h
        gentorch/trace.h
        gentorch/dml/dml.h
        gentorch/distributions/distributions.h
        gentorch/distributions/normal.h
        gentorch/distributions/bernoulli.h
        gentorch/distributions/uniform_continuous.h
        "${CMAKE_CURRENT_SOURCE_DIR}/mainpage.md"
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/include")

# TODO add sphinx on top of Doxygen, as described here:
# https://devblogs.microsoft.com/cppblog/clear-functional-c-documentation-with-sphinx-breathe-doxygen-cmake/
