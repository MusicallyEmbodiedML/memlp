#ifndef __MLP_PLACEMENT_H__
#define __MLP_PLACEMENT_H__

#include <array>
#include <cstddef>
#include <type_traits>

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

// The innermost streaming kernels (StaticLayer::forward / AccumulateGradients).
// A host that owns a scratch SRAM bank (unstriped, outside the banked groups)
// binds this to a section in that bank so the kernels' instruction fetch cannot
// collide with the data they stream (plan-mlpBetterOptimisation.md, Step 5).
// Falls back to SMLP_CODE_ATTR, so a host that does nothing special is unchanged.
#ifndef SMLP_HOT_CODE_ATTR
#define SMLP_HOT_CODE_ATTR SMLP_CODE_ATTR
#endif

// ════════════════════════════════════════════════════════════════════════
//  Bank-phase layout (plan-mlpBetterOptimisation.md, Step 1)
// ════════════════════════════════════════════════════════════════════════
// A bank group stripes consecutive 32-bit words over four banks, so the bank an
// access hits is address bits [3:2], i.e. (address mod 16). How much a hot loop
// collides with its own data depends on (loop head - stream) mod 16. Every
// streamed array therefore sits at a chosen offset from a 16-byte boundary:
//
//   SMLP_WEIGHT_PHASE  byte offset (0, 4, 8 or 12) of the weight, gradient-
//                      accumulator and squared-gradient arrays;
//   SMLP_ACT_PHASE     the same for every activation / scratch stream.
//
// Both are compile-time constants set by the host build; any other value fails
// to compile. The host must also 16-byte-align the instance that owns the
// network, so the phases hold in the absolute address space.
#ifndef SMLP_WEIGHT_PHASE
#define SMLP_WEIGHT_PHASE 0
#endif
#ifndef SMLP_ACT_PHASE
#define SMLP_ACT_PHASE 0
#endif

namespace smlp {

inline constexpr std::size_t kWeightPhase = SMLP_WEIGHT_PHASE;
inline constexpr std::size_t kActPhase    = SMLP_ACT_PHASE;
static_assert(kWeightPhase == 0 || kWeightPhase == 4 || kWeightPhase == 8 || kWeightPhase == 12,
              "SMLP_WEIGHT_PHASE must be 0, 4, 8 or 12 (bytes)");
static_assert(kActPhase == 0 || kActPhase == 4 || kActPhase == 8 || kActPhase == 12,
              "SMLP_ACT_PHASE must be 0, 4, 8 or 12 (bytes)");

namespace detail {
template<std::size_t Bytes> struct PhasePad { unsigned char pad[Bytes]; };
template<> struct PhasePad<0> {};

/// std::array<T, N> whose first element sits `PhaseBytes` past a 16-byte
/// boundary, with the object padded to a multiple of 16 bytes so the next
/// member starts on a boundary again. The pad is the first base, so the array
/// base follows it at offset PhaseBytes. An ordinary std::array otherwise.
template<typename T, std::size_t N, std::size_t PhaseBytes>
struct alignas(16) PhasedArrayImpl : PhasePad<PhaseBytes>, std::array<T, N> {};
} // namespace detail

template<typename T, std::size_t N, std::size_t PhaseBytes>
using PhasedArray = std::conditional_t<(N > 0), detail::PhasedArrayImpl<T, N, PhaseBytes>, std::array<T, 0>>;

template<typename T, std::size_t N> using WeightArray = PhasedArray<T, N, kWeightPhase>;
template<typename T, std::size_t N> using ActArray    = PhasedArray<T, N, kActPhase>;

} // namespace smlp

#endif // __MLP_PLACEMENT_H__
