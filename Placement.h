#ifndef __MLP_PLACEMENT_H__
#define __MLP_PLACEMENT_H__

#ifndef SMLP_CODE_ATTR
#define SMLP_CODE_ATTR
#endif

// For hot-path function TEMPLATES that are instantiated multiple times with
// genuinely different template arguments in one translation unit (e.g. a
// recursive tuple-iteration helper templated on both a layer index and a
// caller-supplied closure type). Deliberately bound to `always_inline`
// rather than a `section` attribute: a `section` attribute is attached to
// the template's ONE textual definition, so any macro built on __COUNTER__
// would fix its section name once per translation unit -- shared by every
// instantiation (every closure type, every recursion depth, and even the
// same instantiation re-emitted from a different TU) -- reintroducing the
// COMDAT-group-per-section-name folding hazard MEML_RUNS_ON_CORE(n) works
// around for ordinary (non-template) functions. `always_inline` sidesteps
// that entirely: these helpers are unconditionally called by every
// genuinely-used caller and only ever call the next recursion depth (a
// distinct instantiation, not real recursion), so GCC can always fully
// inline the call chain, leaving no standalone symbol that would need its
// own placement. See MemoryDefs.hpp's MEML_MLP_CODE_MULTI binding.
#ifndef SMLP_CODE_ATTR_MULTI
#define SMLP_CODE_ATTR_MULTI
#endif

#ifndef SMLP_DATA_ATTR
#define SMLP_DATA_ATTR
#endif

#endif // __MLP_PLACEMENT_H__
