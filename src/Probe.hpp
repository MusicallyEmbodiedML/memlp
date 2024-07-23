#ifndef _PROBE_HPP_
#define _PROBE_HPP_

extern "C" {
#include <stdio.h>
#if defined(__XS3A__)
#include <xscope.h>
#endif
}
#include <vector>

template<typename num_t>
class Probe
{
public:
    Probe(const unsigned int index) :
        index_(index) {
            printf("Probe number %d initialised.\n", index);
        }
    void log(num_t value);
    void log_vector(std::vector<num_t> values);

protected:
    const unsigned int index_;
};


#endif  // _PROBE_HPP_