if (MSVC)
    # warning level 4 and all warnings as errors
    add_compile_options(/W4 /WX)
else()
    # Pedantic warnings and all warnings as errors
    add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()