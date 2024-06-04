

#include <array>
#include <cmath>
#include <iostream>
#include <vector>


double calc_theta(const double s, const std::array<double, 4>& a) {
    double theta_s = a[3];
    for (int i = 2; i >= 0; --i) {
        theta_s = theta_s * s + a[i];
    }
    return theta_s;
}


double IntegrateBySimpson(const std::vector<double>& func, const double dx, const std::size_t n) {
    double sum = func[0] + func[n];
    for (std::size_t i = 1; i < n; ++i) {
        sum += (i % 2 == 0) ? 2.0 * func[i] : 4.0 * func[i];
    }
    return (dx / 3.0) * sum;
}



std::vector<double> generate_f_s0_sn(const double delta_s, const std::array<double, 4>& a, const std::size_t n) {
    std::vector<double> f_s(n + 1);
    std::vector<double> theta_values(n + 1);
    for (size_t i = 0; i <= n; ++i) {
        double s_i = i * delta_s;
        theta_values[i] = calc_theta(s_i, a);
        f_s[i] = std::cos(theta_values[i]);
    }
    return f_s;
}