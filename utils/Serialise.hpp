#ifndef __SERIALISE_HPP__
#define __SERIALISE_HPP__

#include <cstddef>
#include <cstring>
#include <vector>
#include <cstdint>


class Serialise {

 public:

    template <typename T>
    static size_t FromVector2D(size_t w_head,
            const std::vector< std::vector<T> > &vec,
            std::vector<uint8_t> &buffer) {

        // Get sizes
        size_t n_columns = vec.size();
        size_t n_rows = vec[0].size();
        // Reserve space in vector
        buffer.resize(buffer.size() +
            // New size
            (n_columns * n_rows * sizeof(T) + 2 * sizeof(size_t)));
        // Save sizes
        std::vector<size_t> shape { n_columns, n_rows };
        w_head = _LowLevelWrite(w_head, shape, buffer);
        // Save payload
        for (auto &row : vec) {
            w_head = _LowLevelWrite(w_head, row, buffer);
        }

        return w_head;
    };

    template <typename T>
    static size_t ToVector2D(size_t r_head,
            const std::vector<uint8_t> &buffer,
            std::vector< std::vector<T> > &vec) {

        // Get sizes
        std::vector<size_t> shape(2);
        r_head = _LowLevelRead(r_head, 2, buffer, shape);
        // Load into vector
        size_t n_rows = shape[1];
        std::vector<T> temp_row(n_rows);
        vec.resize(shape[0]);
        for (size_t col = 0; col < shape[0]; col++) {
            r_head = _LowLevelRead(r_head, n_rows, buffer, temp_row);
            vec[col] = temp_row;
        }

        return r_head;
    };

 protected:

    template<typename T>
    static size_t _LowLevelWrite(size_t w_head,
            const std::vector<T> &payload,
            std::vector<uint8_t> &buffer) {

        uint8_t *buf_ptr = buffer.data() + w_head;
        size_t size = payload.size() * sizeof(T);
        // Check memory safety
        size_t total_size_bytes = (w_head + size);
        size_t current_size_bytes = buffer.size();
        if (total_size_bytes > current_size_bytes) {
            buffer.resize(total_size_bytes);
        }
        // Perform the copy
        std::memcpy(
            buf_ptr,
            reinterpret_cast<const uint8_t*>(payload.data()),
            size);
        return w_head + size;
    }

    template<typename T>
    static size_t _LowLevelRead(size_t r_head,
            size_t size,
            const std::vector<uint8_t> &buffer,
            std::vector<T> &payload) {

        const uint8_t *buf_ptr = buffer.data() + r_head;
        // Check memory safety
        payload.resize(size);
        size_t size_bytes = size * sizeof(T);
        // Perform the copy
        std::memcpy(
            reinterpret_cast<uint8_t*>(payload.data()),
            buf_ptr,
            size_bytes);
        return r_head + size_bytes;
    }
};


#endif  // __SERIALISE_HPP__