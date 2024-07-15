#include "Probe.hpp"

template <>
void Probe<int>::log(int value)
{
    xscope_int(index_, value);
}


template <>
void Probe<float>::log(float value)
{
    printf(".");
    xscope_float(index_, value);
}


template<class num_t> void Probe<num_t>::log_vector(std::vector<num_t> values)
{
    for (const num_t& value : values) {
        log(value);
    }
    printf("\n");
}


template class Probe<int>;
template class Probe<float>;
