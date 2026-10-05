set(TARGET llama-dflash-dump)
add_executable(${TARGET} dflash-dump.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

add_executable(llama-dflash-parity dflash-parity.cpp)
target_link_libraries(llama-dflash-parity PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(llama-dflash-parity PRIVATE cxx_std_17)
