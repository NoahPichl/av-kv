#include <cfloat>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>


double calc_final_speed(const double v_i, const double a, const double d) {
    double discriminant = v_i * v_i + 2 * a * d;
    double v_f = (discriminant <= 0.0) ? 0.0 : std::sqrt(discriminant);
    return v_f;
}

double calc_distance(const double v_i, const double v_f, const double a) {
    if (std::abs(a) < DBL_EPSILON) {
        return std::numeric_limits<double>::infinity();
    }
    double d = (v_f * v_f - v_i * v_i) / (2 * a);
    return d;
}