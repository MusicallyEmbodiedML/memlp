#include "Probe.hpp"

template <>
void Probe<int>::log(int value)
{
#if defined(__XS3A__)
    xscope_int(index_, value);
#endif
}


template <>
void Probe<float>::log(float value)
{
#if defined(__XS3A__)
    xscope_float(index_, value);
#endif
}


template<class num_t> void Probe<num_t>::log_vector(std::vector<num_t> values)
{
    for (const num_t& value : values) {
        log(value);
    }
}


template class Probe<int>;
template class Probe<float>;
