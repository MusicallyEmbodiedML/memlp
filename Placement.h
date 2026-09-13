#ifndef __MLP_PLACEMENT_H__
#define __MLP_PLACEMENT_H__

#ifndef SMLP_CODE_ATTR
#define SMLP_CODE_ATTR
#endif

// Same placement as SMLP_CODE_ATTR, for hot-path function TEMPLATES that are
// instantiated multiple times with genuinely different template arguments in
// one translation unit (e.g. a recursive tuple-iteration helper templated on
// both a layer index and a caller-supplied closure type). SMLP_CODE_ATTR's
// __COUNTER__-based section name is fixed once at the template's definition
// site, so every such instantiation shares one literal section name; without
// `used`, some optimizers (observed with GCC -O2/-O3) classify different
// instantiations' one-only/comdat linkage inconsistently, which the same
// shared section name then reports as a hard "section type conflict" at
// compile time. `used` is safe here specifically because these helpers are
// unconditionally called by every genuinely-used caller, so it does not
// invite the eager over-instantiation problem `used` causes on class-template
// member functions that may otherwise go uncalled (see SMLP_CODE_ATTR's
// binding in MemoryDefs.hpp).
#ifndef SMLP_CODE_ATTR_MULTI
#define SMLP_CODE_ATTR_MULTI
#endif

#ifndef SMLP_DATA_ATTR
#define SMLP_DATA_ATTR
#endif

#endif // __MLP_PLACEMENT_H__
