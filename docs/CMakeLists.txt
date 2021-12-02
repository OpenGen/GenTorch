set(DOXYGEN_EXTRACT_ALL YES)
set(DOXYGEN_BUILTIN_STL_SUPPORT YES)

doxygen_add_docs(docs
        gen/address.h
        gen/trie.h
        gen/trace.h
        gen/dml.h
        gen/normal.h
        "${CMAKE_CURRENT_SOURCE_DIR}/mainpage.md"
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/include")

# TODO add sphinx on top of Doxygen, as described here:
# https://devblogs.microsoft.com/cppblog/clear-functional-c-documentation-with-sphinx-breathe-doxygen-cmake/