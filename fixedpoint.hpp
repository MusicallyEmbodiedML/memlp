#ifndef FIXEDPOINT_HPP
#define FIXEDPOINT_HPP

#pragma once

#include <cstdint>
#include <concepts>
#include <type_traits>
#include <limits>

// The interpolator-accelerated multiply helpers below are opt-in: they pull in
// hardware/interp.h (and so require linking hardware_interp). Define
// FIXEDPOINT_USE_HW_INTERP before including this header to enable them. Left off
// by default so consumers that don't use the HW multiply path (e.g. the
// fixed-point StaticMLP, which uses a plain 32-bit integer MAC) need no extra link.
#if defined(PICO_SDK_VERSION_MAJOR) && defined(FIXEDPOINT_USE_HW_INTERP)
#include "hardware/interp.h"
#endif

// ============================================================================
// COMPLETE FIXED POINT LIBRARY FOR RP2040
// ============================================================================

namespace FixedPoint {

// ============================================================================
// CONCEPTS
// ============================================================================

template<typename T>
concept IntegerStorage = std::is_integral_v<T> && std::is_signed_v<T>;

template<typename T>
concept FixedPointType = requires {
    typename T::storage_type;
    { T::INTEGER_BITS } -> std::convertible_to<int>;
    { T::FRACTIONAL_BITS } -> std::convertible_to<int>;
};

// ============================================================================
// MAIN FIXED POINT CLASS
// ============================================================================

template<int IntBits, int FracBits, IntegerStorage StorageT = int32_t>
class Fixed {
public:
    using storage_type = StorageT;
    static constexpr int INTEGER_BITS = IntBits;
    static constexpr int FRACTIONAL_BITS = FracBits;
    static constexpr int TOTAL_BITS = IntBits + FracBits;
    static constexpr StorageT ONE = StorageT(1) << FRACTIONAL_BITS;
    static constexpr StorageT HALF = ONE >> 1;
    
    static_assert(std::is_integral<StorageT>::value, "Storage type must be integral");
    static_assert(TOTAL_BITS <= sizeof(StorageT) * 8, "Total bits exceeds storage size");
    static_assert(sizeof(StorageT) <= 4, "Only 32-bit or smaller storage supported");
    
    storage_type value;
    
    // ========================================================================
    // CONSTRUCTORS
    // ========================================================================
    
    constexpr Fixed() : value(0) {}
    
    constexpr explicit Fixed(storage_type raw_value) : value(raw_value) {}
    
    template<std::integral T>
    constexpr explicit Fixed(T integer_value) 
        : value(static_cast<storage_type>(integer_value) << FRACTIONAL_BITS) {}
    
    constexpr explicit Fixed(float f) 
        : value(static_cast<storage_type>(f * ONE + (f >= 0.0f ? 0.5f : -0.5f))) {}
    
    constexpr explicit Fixed(double d) 
        : value(static_cast<storage_type>(d * ONE + (d >= 0.0 ? 0.5 : -0.5))) {}
    

    // Constructor from another Fixed format
    template<int IntBits2, int FracBits2, IntegerStorage StorageT2>
    constexpr explicit Fixed(const Fixed<IntBits2, FracBits2, StorageT2>& other) {
        constexpr int shift_diff = FRACTIONAL_BITS - FracBits2;
        
        if constexpr (shift_diff > 0) {
            // Upscaling - shift left
            value = static_cast<storage_type>(other.raw()) << shift_diff;
        } else if constexpr (shift_diff < 0) {
            // Downscaling - shift right with rounding
            constexpr auto round = StorageT2(1) << (-shift_diff - 1);
            value = static_cast<storage_type>((other.raw() + round) >> -shift_diff);
        } else {
            // Same fractional bits - just cast
            value = static_cast<storage_type>(other.raw());
        }
    }        
    // ========================================================================
    // STATIC FACTORY METHODS
    // ========================================================================
    
    static constexpr Fixed from_int(int32_t i) {
        return Fixed(static_cast<storage_type>(i) << FRACTIONAL_BITS);
    }
    
    static constexpr Fixed from_size_t(size_t i) {
        return Fixed(static_cast<storage_type>(i) << FRACTIONAL_BITS);
    }
    
    static constexpr Fixed from_raw(storage_type raw) {
        return Fixed(raw);
    }
    
    static constexpr Fixed from_fraction(int32_t num, int32_t den) {
        return Fixed((static_cast<int64_t>(num) << FRACTIONAL_BITS) / den);
    }
    
    static consteval Fixed from_float_ct(double f) {
        return Fixed(static_cast<storage_type>(f * ONE + (f >= 0.0 ? 0.5 : -0.5)));
    }
    
    // ========================================================================
    // CONVERSIONS
    // ========================================================================
    
    constexpr int32_t to_int() const {
        return static_cast<int32_t>(value >> FRACTIONAL_BITS);
    }
    
    constexpr int32_t to_int_round() const {
        return static_cast<int32_t>((value + HALF) >> FRACTIONAL_BITS);
    }
    
    constexpr float to_float() const {
        return static_cast<float>(value) / ONE;
    }
    
    constexpr double to_double() const {
        return static_cast<double>(value) / ONE;
    }
    
    constexpr storage_type raw() const {
        return value;
    }
    
    // ========================================================================
    // SAFE ARITHMETIC OPERATORS (USE 64-BIT INTERMEDIATE)
    // ========================================================================
    
    constexpr Fixed operator-() const {
        return Fixed(-value);
    }
    
    constexpr Fixed operator+(const Fixed& other) const {
        return Fixed(value + other.value);
    }
    
    constexpr Fixed operator-(const Fixed& other) const {
        return Fixed(value - other.value);
    }
    
    constexpr Fixed operator*(const Fixed& other) const {
        int64_t result = static_cast<int64_t>(value) * other.value;
        return Fixed(static_cast<storage_type>(result >> FRACTIONAL_BITS));
    }
    
    constexpr Fixed operator/(const Fixed& other) const {
        int64_t result = (static_cast<int64_t>(value) << FRACTIONAL_BITS) / other.value;
        return Fixed(static_cast<storage_type>(result));
    }
    
    constexpr Fixed& operator+=(const Fixed& other) {
        value += other.value;
        return *this;
    }
    
    constexpr Fixed& operator-=(const Fixed& other) {
        value -= other.value;
        return *this;
    }
    
    constexpr Fixed& operator*=(const Fixed& other) {
        *this = *this * other;
        return *this;
    }
    
    constexpr Fixed& operator/=(const Fixed& other) {
        *this = *this / other;
        return *this;
    }

    // ========================================================================
    // SPLIT MULTIPLY - Fast on M0+, avoids 64-bit math
    // ========================================================================

    // Split multiply: fast alternative to mul_fast for M0+ when ranges allow
    // Requirements: 
    //   - value * (other >> FRACTIONAL_BITS) must fit in storage_type
    //   - value * (other & FRAC_MASK) must fit in storage_type
    constexpr Fixed mul_split(const Fixed& other) const {
        constexpr storage_type FRAC_MASK = (storage_type(1) << FRACTIONAL_BITS) - 1;
        
        // Split other into integer and fractional parts
        const storage_type other_int = other.value >> FRACTIONAL_BITS;
        const storage_type other_frac = other.value & FRAC_MASK;
        
        // Compute: result = value * other_int + (value * other_frac) >> FRACTIONAL_BITS
        // Two 32-bit multiplies instead of one 64-bit multiply
        return Fixed::from_raw(
            value * other_int + 
            ((value * other_frac) >> FRACTIONAL_BITS)
        );
    }

    // Split multiply with different format - keeps this format
    template<int IntBits2, int FracBits2, typename StorageT2>
    constexpr Fixed mulWith_split(const Fixed<IntBits2, FracBits2, StorageT2>& other) const {
        constexpr auto FRAC_MASK = (StorageT2(1) << FracBits2) - 1;
        
        const auto other_int = other.raw() >> FracBits2;
        const auto other_frac = other.raw() & FRAC_MASK;
        
        return Fixed::from_raw(
            value * other_int + 
            ((static_cast<int64_t>(value) * other_frac) >> FracBits2)
        );
    }    
    
    // ========================================================================
    // COMPARISON OPERATORS
    // ========================================================================
    
    constexpr bool operator==(const Fixed& other) const { return value == other.value; }
    constexpr bool operator!=(const Fixed& other) const { return value != other.value; }
    constexpr bool operator<(const Fixed& other) const { return value < other.value; }
    constexpr bool operator<=(const Fixed& other) const { return value <= other.value; }
    constexpr bool operator>(const Fixed& other) const { return value > other.value; }
    constexpr bool operator>=(const Fixed& other) const { return value >= other.value; }
    
    // ========================================================================
    // BIT SHIFT OPERATORS
    // ========================================================================
    
    constexpr Fixed operator<<(int shift) const {
        return Fixed(value << shift);
    }
    
    constexpr Fixed operator>>(int shift) const {
        return Fixed(value >> shift);
    }
    
    constexpr Fixed& operator<<=(int shift) {
        value <<= shift;
        return *this;
    }
    
    constexpr Fixed& operator>>=(int shift) {
        value >>= shift;
        return *this;
    }

    // ========================================================================
    // BITWISE OPERATORS
    // ========================================================================

    // Bitwise AND with another Fixed
    constexpr Fixed operator&(const Fixed& other) const {
        return Fixed::from_raw(value & other.value);
    }

    // Bitwise AND with raw integer type
    constexpr Fixed operator&(storage_type mask) const {
        return Fixed::from_raw(value & mask);
    }

    // Bitwise OR with another Fixed
    constexpr Fixed operator|(const Fixed& other) const {
        return Fixed::from_raw(value | other.value);
    }

    // Bitwise OR with raw integer type
    constexpr Fixed operator|(storage_type mask) const {
        return Fixed::from_raw(value | mask);
    }

    // Bitwise XOR with another Fixed
    constexpr Fixed operator^(const Fixed& other) const {
        return Fixed::from_raw(value ^ other.value);
    }

    // Bitwise XOR with raw integer type
    constexpr Fixed operator^(storage_type mask) const {
        return Fixed::from_raw(value ^ mask);
    }

    // Bitwise NOT
    constexpr Fixed operator~() const {
        return Fixed::from_raw(~value);
    }

    // Compound assignment operators
    constexpr Fixed& operator&=(const Fixed& other) {
        value &= other.value;
        return *this;
    }

    constexpr Fixed& operator&=(storage_type mask) {
        value &= mask;
        return *this;
    }

    constexpr Fixed& operator|=(const Fixed& other) {
        value |= other.value;
        return *this;
    }

    constexpr Fixed& operator|=(storage_type mask) {
        value |= mask;
        return *this;
    }

    constexpr Fixed& operator^=(const Fixed& other) {
        value ^= other.value;
        return *this;
    }

    constexpr Fixed& operator^=(storage_type mask) {
        value ^= mask;
        return *this;
    }    

    // Test if specific bits are set
    constexpr bool test_bit(int bit_position) const {
        return (value & (storage_type(1) << bit_position)) != 0;
    }

    // Set a specific bit
    constexpr Fixed set_bit(int bit_position) const {
        return Fixed::from_raw(value | (storage_type(1) << bit_position));
    }

    // Clear a specific bit
    constexpr Fixed clear_bit(int bit_position) const {
        return Fixed::from_raw(value & ~(storage_type(1) << bit_position));
    }

    // Extract bits in a range [low, high]
    constexpr storage_type extract_bits(int low, int high) const {
        int width = high - low + 1;
        storage_type mask = (storage_type(1) << width) - 1;
        return (value >> low) & mask;
    }    
    
    // ========================================================================
    // FAST ARITHMETIC (NO OVERFLOW PROTECTION)
    // ========================================================================
    
    constexpr Fixed mul_fast(const Fixed& other) const {
        return Fixed((value * other.value) >> FRACTIONAL_BITS);
    }
    
    constexpr Fixed div_fast(const Fixed& other) const {
        return Fixed((value << FRACTIONAL_BITS) / other.value);
    }
    
    constexpr Fixed mul_fast_shift(const Fixed& other, int result_shift) const {
        return Fixed((value * other.value) >> result_shift);
    }
    
    constexpr Fixed mul_int(int32_t i) const {
        return Fixed(value * i);
    }
    
    constexpr Fixed div_int(int32_t i) const {
        return Fixed(value / i);
    }
    
    constexpr Fixed mul_pow2(int shift) const {
        return Fixed(value << shift);
    }
    
    constexpr Fixed div_pow2(int shift) const {
        return Fixed(value >> shift);
    }

    constexpr Fixed safeShiftLeft(int n) {
        const int32_t limit = INT32_MAX >> n;       // max value that survives << n
        if (value >  limit) return Fixed(INT32_MAX);
        if (value < ~limit) return Fixed(INT32_MIN);
        return Fixed(value << n);
    }


    // Multiply with another Fixed type, keeping this format
    template<int IntBits2, int FracBits2, typename StorageT2>
    constexpr Fixed<IntBits, FracBits, StorageT> mulWith(
        const Fixed<IntBits2, FracBits2, StorageT2>& other
    ) const {
        // Always use int64_t for 32-bit storage multiplication
        using WideT = int64_t;
        
        // Perform multiplication in wider type
        WideT temp = static_cast<WideT>(value) * static_cast<WideT>(other.raw());
        
        // Shift by other's fractional bits
        WideT shifted = temp >> FracBits2;
                
        return Fixed<IntBits, FracBits, StorageT>::from_raw(
            static_cast<StorageT>(shifted)
        );
    }

    // Divide by another Fixed type, keeping this format
    template<int IntBits2, int FracBits2, typename StorageT2>
    constexpr Fixed<IntBits, FracBits, StorageT> divWith(
        const Fixed<IntBits2, FracBits2, StorageT2>& other
    ) const {
        // Always use int64_t for 32-bit storage division
        using WideT = int64_t;
        
        // Shift dividend left by divisor's fractional bits to maintain precision
        WideT temp = static_cast<WideT>(value) << FracBits2;
        
        // Perform division in wider type
        WideT result = temp / static_cast<WideT>(other.raw());
                
        return Fixed<IntBits, FracBits, StorageT>::from_raw(
            static_cast<StorageT>(result)
        );
    }    

    constexpr Fixed frac() const {
        // Mask to keep only fractional bits: frac(x) = x - floor(x)
        storage_type mask = (storage_type(1) << FRACTIONAL_BITS) - 1;
        return Fixed::from_raw(value & mask);
    }    

    constexpr Fixed frac_signed() const {
        return *this - Fixed::from_int(to_int());
    }    

    // fmod(x, y) = x - trunc(x/y) * y  (result sign matches x)
    __force_inline constexpr Fixed fmod(const Fixed& divisor) const {
        Fixed quotient = *this / divisor;
        // Truncate toward zero (arithmetic shift rounds toward -inf, fix for negatives)
        storage_type int_part = quotient.value >> FRACTIONAL_BITS;
        constexpr storage_type frac_mask = (storage_type(1) << FRACTIONAL_BITS) - 1;
        if (quotient.value < 0 && (quotient.value & frac_mask)) {
            int_part += 1;
        }
        return *this - Fixed::from_raw(int_part << FRACTIONAL_BITS) * divisor;
    }    
    
    // ========================================================================
    // SATURATING ARITHMETIC
    // ========================================================================
    
    constexpr Fixed add_sat(const Fixed& other) const {
        int64_t result = static_cast<int64_t>(value) + other.value;
        constexpr int64_t max_val = std::numeric_limits<storage_type>::max();
        constexpr int64_t min_val = std::numeric_limits<storage_type>::min();
        if (result > max_val) return Fixed(static_cast<storage_type>(max_val));
        if (result < min_val) return Fixed(static_cast<storage_type>(min_val));
        return Fixed(static_cast<storage_type>(result));
    }
    
    constexpr Fixed sub_sat(const Fixed& other) const {
        int64_t result = static_cast<int64_t>(value) - other.value;
        constexpr int64_t max_val = std::numeric_limits<storage_type>::max();
        constexpr int64_t min_val = std::numeric_limits<storage_type>::min();
        if (result > max_val) return Fixed(static_cast<storage_type>(max_val));
        if (result < min_val) return Fixed(static_cast<storage_type>(min_val));
        return Fixed(static_cast<storage_type>(result));
    }
    
    constexpr Fixed mul_sat(const Fixed& other) const {
        int64_t result = (static_cast<int64_t>(value) * other.value) >> FRACTIONAL_BITS;
        constexpr int64_t max_val = std::numeric_limits<storage_type>::max();
        constexpr int64_t min_val = std::numeric_limits<storage_type>::min();
        if (result > max_val) return Fixed(static_cast<storage_type>(max_val));
        if (result < min_val) return Fixed(static_cast<storage_type>(min_val));
        return Fixed(static_cast<storage_type>(result));
    }
    
    // ========================================================================
    // RECIPROCAL APPROXIMATION
    // ========================================================================
    
    inline Fixed reciprocal_fast() const {
        if (value == 0) return Fixed::from_int(0);
        
        storage_type x = value;
        storage_type sign = x >> 31;
        if (sign) x = -x;
        
        int lz = __builtin_clz(x);
        storage_type guess = (storage_type(1) << (31 - lz)) >> (FRACTIONAL_BITS - lz);
        
        // Newton-Raphson iterations
        for (int i = 0; i < 3; i++) {
            storage_type temp = (static_cast<int64_t>(x) * guess) >> FRACTIONAL_BITS;
            temp = (ONE << 1) - temp;
            guess = (static_cast<int64_t>(guess) * temp) >> FRACTIONAL_BITS;
        }
        
        return Fixed(sign ? -guess : guess);
    }
    
    inline Fixed div_by_recip_fast(const Fixed& denominator) const {
        Fixed recip = denominator.reciprocal_fast();
        return mul_fast(recip);
    }

    // ========================================================================
    // Random numbers
    // ========================================================================

    // Hardware RNG - optimized version
    static Fixed random_hw(const Fixed& min_val, const Fixed& max_val) {
        storage_type range = max_val.value - min_val.value;
        
        if (range == 0) return min_val;
        
        uint32_t r = get_rand_32();
        
        // For small ranges, avoid 64-bit math
        // Safe if range * (r >> 16) fits in 32 bits
        if (range < (1 << 15)) {
            // Split r into high and low parts for better distribution
            storage_type scaled = ((r >> 16) * range) >> 16;
            return Fixed::from_raw(min_val.value + scaled);
        } else {
            // Need 64-bit for larger ranges
            int64_t scaled = (static_cast<int64_t>(r) * range) >> 32;
            return Fixed::from_raw(min_val.value + static_cast<storage_type>(scaled));
        }
    }

    // Random, using c++ lib
    static Fixed random(const Fixed& min_val, const Fixed& max_val) {
        storage_type range = max_val.value - min_val.value;
        
        if (range == 0) return min_val;
        
        uint32_t r = rand();
        
        // For small ranges, avoid 64-bit math
        // Safe if range * (r >> 16) fits in 32 bits
        if (range < (1 << 15)) {
            // Split r into high and low parts for better distribution
            storage_type scaled = ((r >> 16) * range) >> 16;
            return Fixed::from_raw(min_val.value + scaled);
        } else {
            // Need 64-bit for larger ranges
            int64_t scaled = (static_cast<int64_t>(r) * range) >> 32;
            return Fixed::from_raw(min_val.value + static_cast<storage_type>(scaled));
        }
    }
    // Fast [0, 1) range using hardware RNG
    static Fixed random_unit_hw() {
        uint32_t r = get_rand_32();
        // Shift to fit our fractional bits
        return Fixed::from_raw(static_cast<storage_type>(r >> (32 - FRACTIONAL_BITS)));
    }

    // Fast [-1, 1) range using hardware RNG  
    static Fixed random_bipolar_hw() {
        int32_t r = static_cast<int32_t>(get_rand_32());
        return Fixed::from_raw(r >> (32 - FRACTIONAL_BITS - 1));
    }
    
    // ========================================================================
    // RP2040 HARDWARE ACCELERATED OPERATIONS
    // ========================================================================
    
    #if defined(PICO_SDK_VERSION_MAJOR) && defined(FIXEDPOINT_USE_HW_INTERP)

    inline Fixed mul_hw_interp(const Fixed& other) const {
        interp_config cfg = interp_default_config();
        interp_config_set_shift(&cfg, FRACTIONAL_BITS);
        interp_config_set_signed(&cfg, true);
        interp_set_config(interp0, 0, &cfg);
        
        interp0->accum[0] = value;
        interp0->base[0] = other.value;
        
        return Fixed(static_cast<storage_type>(interp0->peek[0]));
    }
    
    inline Fixed mul_add_hw(const Fixed& other, const Fixed& accum) const {
        interp_config cfg = interp_default_config();
        interp_config_set_shift(&cfg, FRACTIONAL_BITS);
        interp_config_set_signed(&cfg, true);
        interp_config_set_add_raw(&cfg, true);
        interp_set_config(interp0, 0, &cfg);
        
        interp0->accum[0] = accum.value;
        interp0->base[0] = other.value;
        interp0->base[1] = value;
        
        return Fixed(static_cast<storage_type>(interp0->peek[0]));
    }
    
    static void mul_batch_hw(const Fixed* a, const Fixed* b, Fixed* result, size_t count) {
        interp_config cfg = interp_default_config();
        interp_config_set_shift(&cfg, FRACTIONAL_BITS);
        interp_config_set_signed(&cfg, true);
        interp_set_config(interp0, 0, &cfg);
        
        for (size_t i = 0; i < count; i++) {
            interp0->accum[0] = a[i].value;
            interp0->base[0] = b[i].value;
            result[i] = Fixed(static_cast<storage_type>(interp0->peek[0]));
        }
    }
    
    #endif // PICO_SDK_VERSION_MAJOR
};

// ============================================================================
// TYPE ALIASES
// ============================================================================

using Q15_16 = Fixed<15, 16, int32_t>;
using Q19_13 = Fixed<19, 13, int32_t>;
using Q7_24  = Fixed<7, 24, int32_t>;
using Q1_31  = Fixed<1, 31, int32_t>;
using Q2_30  = Fixed<2, 30, int32_t>;
using Q16_16 = Fixed<16, 16, int32_t>;
using Q8_24  = Fixed<8, 24, int32_t>;
using Q12_20 = Fixed<12, 20, int32_t>;

// ============================================================================
// FORMAT CONVERSION
// ============================================================================

template<int ToIntBits, int ToFracBits, typename ToStorage,
         int FromIntBits, int FromFracBits, typename FromStorage>
constexpr Fixed<ToIntBits, ToFracBits, ToStorage> 
convert(const Fixed<FromIntBits, FromFracBits, FromStorage>& from) {
    constexpr int shift_diff = ToFracBits - FromFracBits;
    
    if constexpr (shift_diff > 0) {
        return Fixed<ToIntBits, ToFracBits, ToStorage>::from_raw(
            static_cast<ToStorage>(from.raw()) << shift_diff
        );
    } else if constexpr (shift_diff < 0) {
        constexpr auto round = FromStorage(1) << (-shift_diff - 1);
        return Fixed<ToIntBits, ToFracBits, ToStorage>::from_raw(
            static_cast<ToStorage>((from.raw() + round) >> -shift_diff)
        );
    } else {
        return Fixed<ToIntBits, ToFracBits, ToStorage>::from_raw(
            static_cast<ToStorage>(from.raw())
        );
    }
}



// ============================================================================
// BASIC MATH FUNCTIONS
// ============================================================================

template<FixedPointType T>
__force_inline constexpr T abs(const T& x) {
    return (x.value < 0) ? T::from_raw(-x.value) : x;
}

template<FixedPointType T>
__force_inline constexpr T min(const T& a, const T& b) {
    return (a.value < b.value) ? a : b;
}

template<FixedPointType T>
__force_inline constexpr T max(const T& a, const T& b) {
    return (a.value > b.value) ? a : b;
}

template<FixedPointType T>
constexpr T clamp(const T& value, const T& min_val, const T& max_val) {
    return min(max(value, min_val), max_val);
}

template<FixedPointType T>
__force_inline constexpr T floor(const T& x) {
    // Clear fractional bits using shift operations instead of mask
    // This avoids loading a large constant from flash
    return T::from_raw((x.value >> T::FRACTIONAL_BITS) << T::FRACTIONAL_BITS);
}

template<FixedPointType T>
__force_inline constexpr T ceil(const T& x) {
    // Extract integer part by shifting
    typename T::storage_type int_part = x.value >> T::FRACTIONAL_BITS;
    typename T::storage_type reconstructed = int_part << T::FRACTIONAL_BITS;
    
    // Check if there's a fractional part
    if (x.value == reconstructed) {
        return x;  // No fractional part, already at ceiling
    }
    
    // Has fractional part: ceil is always floor + 1 (for both positive and negative)
    // For positive: -2.3 has floor=-3, ceil=-2 (floor + 1)
    // For negative: 2.3 has floor=2, ceil=3 (floor + 1)
    return T::from_raw((int_part + 1) << T::FRACTIONAL_BITS);
}

template<FixedPointType T>
__force_inline constexpr T round(const T& x) {
    // Add 0.5 using shift (1 << (FRACTIONAL_BITS - 1))
    // Then floor by shift round-trip
    // This avoids loading T::HALF from memory
    typename T::storage_type half_bit = typename T::storage_type(1) << (T::FRACTIONAL_BITS - 1);
    typename T::storage_type with_half = x.value + half_bit;
    return T::from_raw((with_half >> T::FRACTIONAL_BITS) << T::FRACTIONAL_BITS);
}

template<FixedPointType T>
__force_inline constexpr T fmod(const T& x, const T& divisor) {
    return x.fmod(divisor);
}

template<FixedPointType T>
inline T sqrt(const T& x) {
    if (x.value <= 0) return T::from_int(0);
    
    typename T::storage_type val = x.value;
    typename T::storage_type guess = val >> 1;
    
    for (int i = 0; i < 8; i++) {
        if (guess == 0) break;
        typename T::storage_type div = (static_cast<int64_t>(val) << T::FRACTIONAL_BITS) / guess;
        guess = (guess + div) >> 1;
    }
    
    return T::from_raw(guess);
}

template<FixedPointType T>
constexpr T lerp(const T& a, const T& b, const T& t) {
    return a + (b - a) * t;
}

// ============================================================================
// TRIGONOMETRY
// ============================================================================

template<FixedPointType T>
inline T sin_fast(const T& x) {
    constexpr auto FIXED_PI = T::from_float_ct(3.14159265359);
    constexpr auto FIXED_TWO_PI = T::from_float_ct(6.28318530718);

    T norm = x;
    while (norm > FIXED_PI) norm -= FIXED_TWO_PI;
    while (norm < -FIXED_PI) norm += FIXED_TWO_PI;
    
    T x2 = norm * norm;
    T num = norm * (T::from_float_ct(16.0) - T::from_float_ct(5.0) * x2);
    T den = T::from_float_ct(16.0) + x2;
    
    return num / den;
}

template<FixedPointType T>
inline T cos_fast(const T& x) {
    constexpr auto FIXED_HALF_PI = T::from_float_ct(1.57079632679);
    return sin_fast(x + FIXED_HALF_PI);
}

// ============================================================================
// FAST MIXED-FORMAT OPERATIONS
// ============================================================================

template<int ResIntBits, int ResFracBits, typename ResStorage,
         int AIntBits, int AFracBits, typename AStorage,
         int BIntBits, int BFracBits, typename BStorage>
constexpr Fixed<ResIntBits, ResFracBits, ResStorage>
mul_fast_mixed(const Fixed<AIntBits, AFracBits, AStorage>& a,
               const Fixed<BIntBits, BFracBits, BStorage>& b) {
    constexpr int shift_amount = AFracBits + BFracBits - ResFracBits;
    auto result = (static_cast<ResStorage>(a.raw()) * b.raw()) >> shift_amount;
    return Fixed<ResIntBits, ResFracBits, ResStorage>::from_raw(result);
}

template<FixedPointType T>
constexpr T madd_fast(const T& a, const T& b, const T& c) {
    auto prod = (b.value * c.value) >> T::FRACTIONAL_BITS;
    return T::from_raw(a.value + prod);
}

template<FixedPointType T>
constexpr T msub_fast(const T& a, const T& b, const T& c) {
    auto prod = (b.value * c.value) >> T::FRACTIONAL_BITS;
    return T::from_raw(a.value - prod);
}

// ============================================================================
// ARRAY/BATCH OPERATIONS
// ============================================================================

template<FixedPointType T>
T dot_product_fast(const T* a, const T* b, size_t count) {
    typename T::storage_type accum = 0;
    for (size_t i = 0; i < count; i++) {
        accum += (a[i].value * b[i].value) >> T::FRACTIONAL_BITS;
    }
    return T::from_raw(accum);
}

template<FixedPointType T>
T poly_eval_fast(const T& x, const T* coeffs, size_t degree) {
    T result = coeffs[degree];
    for (int i = degree - 1; i >= 0; i--) {
        result = result.mul_fast(x) + coeffs[i];
    }
    return result;
}

#if defined(PICO_SDK_VERSION_MAJOR) && defined(FIXEDPOINT_USE_HW_INTERP)

template<FixedPointType T>
void mul_array_hw(const T* a, const T* b, T* result, size_t count) {
    T::mul_batch_hw(a, b, result, count);
}

template<FixedPointType T>
void mul_scalar_hw(const T* input, const T& scalar, T* output, size_t count) {
    interp_config cfg = interp_default_config();
    interp_config_set_shift(&cfg, T::FRACTIONAL_BITS);
    interp_config_set_signed(&cfg, true);
    interp_set_config(interp0, 0, &cfg);
    
    interp0->base[0] = scalar.value;
    
    for (size_t i = 0; i < count; i++) {
        interp0->accum[0] = input[i].value;
        output[i] = T::from_raw(static_cast<typename T::storage_type>(interp0->peek[0]));
    }
}

template<FixedPointType T>
void fma_array_hw(const T* a, const T* b, const T* c, T* result, size_t count) {
    interp_config cfg = interp_default_config();
    interp_config_set_shift(&cfg, T::FRACTIONAL_BITS);
    interp_config_set_signed(&cfg, true);
    interp_config_set_add_raw(&cfg, true);
    interp_set_config(interp0, 0, &cfg);
    
    for (size_t i = 0; i < count; i++) {
        interp0->accum[0] = a[i].value;
        interp0->base[0] = c[i].value;
        interp0->base[1] = b[i].value;
        result[i] = T::from_raw(static_cast<typename T::storage_type>(interp0->peek[0]));
    }
}

#endif // PICO_SDK_VERSION_MAJOR

// ============================================================================
// USER-DEFINED LITERALS
// ============================================================================

namespace literals {
    consteval Q15_16 operator""_q15(long double f) {
        return Q15_16::from_float_ct(static_cast<double>(f));
    }
    
    consteval Q19_13 operator""_q19(long double f) {
        return Q19_13::from_float_ct(static_cast<double>(f));
    }
    
    consteval Q7_24 operator""_q7(long double f) {
        return Q7_24::from_float_ct(static_cast<double>(f));
    }
    
    consteval Q1_31 operator""_q1(long double f) {
        return Q1_31::from_float_ct(static_cast<double>(f));
    }
    
    consteval Q16_16 operator""_q16(long double f) {
        return Q16_16::from_float_ct(static_cast<double>(f));
    }
}

// ============================================================================
// DEBUG/PRINTING SUPPORT
// ============================================================================

#ifdef ARDUINO

template<FixedPointType T>
void print_fixed(const T& x) {
    Serial.print(x.to_float(), 6);
}

template<FixedPointType T>
void println_fixed(const T& x) {
    Serial.println(x.to_float(), 6);
}

template<FixedPointType T>
void print_fixed_raw(const T& x) {
    Serial.print("Raw: 0x");
    Serial.print(x.raw(), HEX);
    Serial.print(" (");
    Serial.print(x.raw());
    Serial.print(") = ");
    Serial.print(x.to_float(), 6);
}

template<FixedPointType T>
void print_fixed_info(const T& x) {
    Serial.print("Format: Q");
    Serial.print(T::INTEGER_BITS);
    Serial.print(".");
    Serial.print(T::FRACTIONAL_BITS);
    Serial.print(", Value: ");
    Serial.print(x.to_float(), 6);
    Serial.print(", Raw: 0x");
    Serial.println(x.raw(), HEX);
}

#endif // ARDUINO


} // namespace FixedPoint

#endif // FIXEDPOINT_HPP
