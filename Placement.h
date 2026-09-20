#ifndef __MLP_PLACEMENT_H__
#define __MLP_PLACEMENT_H__

#ifndef SMLP_CODE_ATTR
#define SMLP_CODE_ATTR
#endif

// For hot-path function TEMPLATES that are instantiated multiple times with
// genuinely different template arguments in one translation unit (e.g. a
// recursive tuple-iteration helper templated on both a layer index and a
// caller-supplied closure type). Deliberately left blank by default here --
// no `section` attribute: one would be attached to the template's ONE
// textual definition, so any macro built on __COUNTER__ would fix its
// section name once per translation unit, shared by every instantiation,
// reintroducing the COMDAT-group-per-section-name folding hazard
// MEML_RUNS_ON_CORE(n) works around for ordinary (non-template) functions.
// Left blank, -ffunction-sections gives each instantiation its own
// auto-named `.text.<mangled-name>` section -- already safely unique per
// symbol, no macro needed -- which lets the compiler keep one small
// out-of-line copy per (closure type, recursion depth) and reuse it via
// `bl`, instead of a full unrolled copy at every call site; measured ~4%
// faster on hardware than forcing `always_inline` everywhere. MemoryDefs.hpp
// overrides this to `always_inline` for ONE specific configuration
// (core-1-pinned) where the auto-named section can't be redirected into
// that core's bank without risking the boot image layout -- see its
// MEML_MLP_CODE_MULTI comment for the full reasoning.
#ifndef SMLP_CODE_ATTR_MULTI
#define SMLP_CODE_ATTR_MULTI
#endif

#ifndef SMLP_DATA_ATTR
#define SMLP_DATA_ATTR
#endif

#endif // __MLP_PLACEMENT_H__
