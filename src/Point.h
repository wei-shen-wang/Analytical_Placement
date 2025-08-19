#ifndef POINT_H
#define POINT_H

#include <algorithm>
#include <cmath>

/**
 * @brief Template class for 2D point
 *
 * @tparam T Type of the point
 */
template <typename T>
struct Point2 {
    /////////////////////////////////
    // Constructors
    /////////////////////////////////

    /// Default constructor
    Point2() : x(0), y(0) {}

    /// Scalar constructor
    explicit Point2(const T &u) : x(u), y(u) {}

    /// Construct from two values
    Point2(const T &u, const T &v) : x(u), y(v) {}

    /////////////////////////////////
    // Assignment operators
    /////////////////////////////////

    /// Copy from scalar
    Point2<T> &operator=(const T &u) {
        x = u;
        y = u;
        return *this;
    }

    /////////////////////////////////
    // Indexing operators
    /////////////////////////////////

    /// Access by index
    T &operator[](const size_t &i) { return array_[i]; }

    /// Access by index
    const T &operator[](const size_t &i) const { return array_[i]; }

    /////////////////////////////////
    // Math operators
    /////////////////////////////////

    /// Scalar add and assign
    Point2<T> &operator+=(const T &rhs) {
        x += rhs;
        y += rhs;
        return *this;
    }

    /// Element-wise add and assign
    Point2<T> &operator+=(const Point2<T> &rhs) {
        x += rhs.x;
        y += rhs.y;
        return *this;
    }

    /// Scalar subtract and assign
    Point2<T> &operator-=(const T &rhs) {
        x -= rhs;
        y -= rhs;
        return *this;
    }

    /// Element-wise subtract and assign
    Point2<T> &operator-=(const Point2<T> &rhs) {
        x -= rhs.x;
        y -= rhs.y;
        return *this;
    }

    /// Scalar multiply and assign
    Point2<T> &operator*=(const T &rhs) {
        x *= rhs;
        y *= rhs;
        return *this;
    }

    /// Element-wise multiply and assign
    Point2<T> &operator*=(const Point2<T> &rhs) {
        x *= rhs.x;
        y *= rhs.y;
        return *this;
    }

    /// Scalar divide and assign
    Point2<T> &operator/=(const T &rhs) {
        x /= rhs;
        y /= rhs;
        return *this;
    }

    /// Element-wise divide and assign
    Point2<T> &operator/=(const Point2<T> &rhs) {
        x /= rhs.x;
        y /= rhs.y;
        return *this;
    }

    /// Unary negate
    Point2<T> operator-() const { return Point2<T>(-x, -y); }

    /// Scalar(right) add
    friend Point2<T> operator+(const Point2<T> lhs, const T &rhs) {
        return Point2<T>(lhs) += rhs;
    }

    /// Scalar(left) add
    friend Point2<T> operator+(const T &lhs, const Point2<T> rhs) {
        return Point2<T>(lhs) += rhs;
    }

    /// Element-wise add
    friend Point2<T> operator+(const Point2<T> lhs, const Point2<T> &rhs) {
        return Point2<T>(lhs) += rhs;
    }

    /// Scalar(right) subtract
    friend Point2<T> operator-(const Point2<T> lhs, const T &rhs) {
        return Point2<T>(lhs) -= rhs;
    }

    /// Scalar(left) subtract
    friend Point2<T> operator-(const T &lhs, const Point2<T> rhs) {
        return Point2<T>(lhs) -= rhs;
    }

    /// Element-wise subtract
    friend Point2<T> operator-(const Point2<T> lhs, const Point2<T> &rhs) {
        return Point2<T>(lhs) -= rhs;
    }

    /// Scalar(right) multiply
    friend Point2<T> operator*(const Point2<T> lhs, const T &rhs) {
        return Point2<T>(lhs) *= rhs;
    }

    /// Scalar(left) multiply
    friend Point2<T> operator*(const T &lhs, const Point2<T> rhs) {
        return Point2<T>(lhs) *= rhs;
    }

    /// Element-wise multiply
    friend Point2<T> operator*(const Point2<T> lhs, const Point2<T> &rhs) {
        return Point2<T>(lhs) *= rhs;
    }

    /// Scalar(right) divide
    friend Point2<T> operator/(const Point2<T> lhs, const T &rhs) {
        return Point2<T>(lhs) /= rhs;
    }

    /// Scalar(left) divide
    friend Point2<T> operator/(const T &lhs, const Point2<T> rhs) {
        return Point2<T>(lhs) /= rhs;
    }

    /// Element-wise divide
    friend Point2<T> operator/(const Point2<T> lhs, const Point2<T> &rhs) {
        return Point2<T>(lhs) /= rhs;
    }

    /////////////////////////////////
    // Comparison operators
    /////////////////////////////////

    /// Equality
    friend bool operator==(const Point2<T> &lhs, const Point2<T> &rhs) {
        return (lhs.x == rhs.x) && (lhs.y == rhs.y);
    }

    /// Inequality
    friend bool operator!=(const Point2<T> &lhs, const Point2<T> &rhs) {
        return !(lhs == rhs);
    }

    /////////////////////////////////
    // Math functions
    /////////////////////////////////

    /// Dot product
    friend T Dot(const Point2<T> &lhs, const Point2<T> &rhs) {
        return lhs.x * rhs.x + lhs.y * rhs.y;
    }

    /// Cross product
    friend T Cross(const Point2<T> &lhs, const Point2<T> &rhs) {
        return lhs.x * rhs.y - lhs.y * rhs.x;
    }

    /// 2-norm
    friend T Norm2(const Point2<T> &u) { return std::sqrt(u.x * u.x + u.y * u.y); }

    /// Element-wise exponential
    friend Point2<T> Exp(const Point2<T> &u) {
        return Point2<T>(std::exp(u.x), std::exp(u.y));
    }

    /// Element-wise minimum
    friend Point2<T> Min(const Point2<T> &u, const Point2<T> &v) {
        return Point2<T>(std::min(u.x, v.x), std::min(u.y, v.y));
    }

    /// Element-wise maximum
    friend Point2<T> Max(const Point2<T> &u, const Point2<T> &v) {
        return Point2<T>(std::max(u.x, v.x), std::max(u.y, v.y));
    }

    /// Element-wise clamp
    friend Point2<T> Clamp(const Point2<T> &u, const Point2<T> &lo, const Point2<T> &hi) {
        return Point2<T>(std::clamp(u.x, lo.x, hi.x), std::clamp(u.y, lo.y, hi.y));
    }

    /// Is finite
    friend bool IsFinite(const Point2<T> &u) {
        return std::isfinite(u.x) && std::isfinite(u.y);
    }

    /////////////////////////////////
    // Data members
    /////////////////////////////////

    // With unnamed union, we can access x and y directly
    // For example:
    //
    // Point2<T> p;
    // p.x = 1; // This will work
    // p.y = 1; // This will work
    // p[0] = 1; // This also will work, which is equivalent to p.x = 1
    // p[1] = 1; // This also will work, which is equivalent to p.y = 1
    //
    // Here, p.x and p[0] shared the same memory location, so does p.y and p[1]
    union {
        struct
        {
            T x;
            T y;
        };
        T array_[2];
    };
};

#endif  // POINT_H
