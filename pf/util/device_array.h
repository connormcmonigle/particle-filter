#pragma once

#include <pf/config/target_config.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>

namespace pf::util {

template <typename T, std::size_t N>
struct device_array {
  using value_type = T;

  T data_[N];

  PF_TARGET_ATTRS [[nodiscard]] constexpr std::size_t size() const noexcept { return N; }

  PF_TARGET_ATTRS [[nodiscard]] constexpr T* data() noexcept { return data_; }
  PF_TARGET_ATTRS [[nodiscard]] constexpr const T* data() const noexcept { return data_; }

  PF_TARGET_ATTRS [[nodiscard]] constexpr T* begin() noexcept { return data_; }
  PF_TARGET_ATTRS [[nodiscard]] constexpr T* end() noexcept { return data_ + N; }

  PF_TARGET_ATTRS [[nodiscard]] constexpr const T* begin() const noexcept { return data_; }
  PF_TARGET_ATTRS [[nodiscard]] constexpr const T* end() const noexcept { return data_ + N; }

  PF_TARGET_ATTRS [[nodiscard]] constexpr const T* cbegin() const noexcept { return data_; }
  PF_TARGET_ATTRS [[nodiscard]] constexpr const T* cend() const noexcept { return data_ + N; }

  PF_TARGET_ATTRS [[nodiscard]] constexpr T& operator[](const std::size_t& idx) noexcept { return data_[idx]; }
  PF_TARGET_ATTRS [[nodiscard]] constexpr const T& operator[](const std::size_t& idx) const noexcept { return data_[idx]; }

  template <std::size_t Index>
  PF_TARGET_ATTRS [[nodiscard]] T& get() noexcept {
    return data_[Index];
  }

  template <std::size_t Index>
  PF_TARGET_ATTRS [[nodiscard]] const T& get() const noexcept {
    return data_[Index];
  }

  [[nodiscard]] std::array<T, N> to_host_array() const noexcept {
    std::array<T, N> host_array{};
    std::copy(data_, data_ + N, host_array.begin());
    return host_array;
  };

  template <typename F>
  PF_TARGET_ATTRS [[nodiscard]] inline device_array<typename std::result_of<F(T)>::type, N> transformed(F&& f) const noexcept {
    device_array<typename std::result_of<F(T)>::type, N> result{};
    auto in_iter = data_;
    for (auto out_iter = result.begin(); out_iter != result.end(); ++in_iter, ++out_iter) { *out_iter = f(*in_iter); }
    return result;
  }

  template <typename F>
  PF_TARGET_ATTRS [[nodiscard]] inline device_array<T, N>& selection_sort_by(F&& f) noexcept {
    auto keys = transformed(f);

    auto i_val = data_;
    for (auto i_key = keys.begin(); i_key != keys.end(); ++i_val, ++i_key) {
      auto min_key = i_key;
      auto min_val = i_val;

      auto j_val = i_val;
      for (auto j_key = i_key; j_key != keys.end(); ++j_val, ++j_key) {
        if (*j_key < *min_key) {
          min_key = j_key;
          min_val = j_val;
        }
      }

      thrust::swap(*i_key, *min_key);
      thrust::swap(*i_val, *min_val);
    }

    return *this;
  }

  template <typename F>
  PF_TARGET_ATTRS [[nodiscard]] inline const T* minimum_by(F&& f) const noexcept {
    const auto keys = transformed(f);

    auto min_val = data_;
    auto min_key = keys.begin();

    auto i_val = data_;
    auto i_key = keys.begin();

    ++i_val;
    ++i_key;

    for (; i_key != keys.end(); ++i_val, ++i_key) {
      if (*i_key < *min_key) {
        min_key = i_key;
        min_val = i_val;
      }
    }

    return min_val;
  }
};

}  // namespace pf::util

namespace std {

template <typename T, size_t N>
struct tuple_size<util::device_array<T, N>> : std::integral_constant<size_t, N> {};

template <size_t Index, typename T, size_t N>
struct tuple_element<Index, util::device_array<T, N>> {
  using type = T;
};

}  // namespace std
