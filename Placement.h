#ifndef __MLP_PLACEMENT_H__
#define __MLP_PLACEMENT_H__

#ifndef SMLP_CODE_ATTR
#define SMLP_CODE_ATTR
#endif

// For hot-path function TEMPLATES that are instantiated multiple times with
// genuinely different template arguments in one translation unit (e.g. a
// recursive tuple-iteration helper templated on both a layer index and a
// caller-supplied closure type). Always blank -- in every configuration,
// including a core-1-pinned host binding: no `section` attribute, since one
// would be attached to the template's ONE textual definition, so any macro
// built on __COUNTER__ would fix its section name once per translation
// unit, shared by every instantiation, reintroducing the COMDAT-group-per-
// section-name folding hazard MEML_RUNS_ON_CORE(n) works around for
// ordinary (non-template) functions. Left blank, -ffunction-sections gives
// each instantiation its own auto-named `.text.<mangled-name>` section --
// already safely unique per symbol, no macro needed -- which lets the
// compiler keep one small out-of-line copy per (closure type, recursion
// depth) and reuse it via `bl`, instead of a full unrolled copy at every
// call site; measured faster on hardware than forcing a fixed structure
// (either `always_inline` everywhere or a `noinline` wrapper at every call
// site) in any configuration. A host project whose target bank for this
// code isn't the toolchain's own default RAM region (e.g. this project's
// core-1-pinned build) redirects those auto-named sections at the linker
// level instead of tagging the template -- see mlp/StaticMLP.h's
// for_each_layer for the full reasoning, and this project's own
// linker/section_copy_to_ram_text.incl for the redirect itself.
#ifndef SMLP_CODE_ATTR_MULTI
#define SMLP_CODE_ATTR_MULTI
#endif

#ifndef SMLP_DATA_ATTR
#define SMLP_DATA_ATTR
#endif

#endif // __MLP_PLACEMENT_H__
