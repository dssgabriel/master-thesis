#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <thread>
#include <vector>

constexpr size_t NELEMENTS = 1'000'000;
constexpr size_t NTHREADS = 4;
constexpr size_t QOT = NELEMENTS / NTHREADS;
constexpr size_t REM = NELEMENTS % NTHREADS;

auto main() -> int {
    std::vector<double> vector(NELEMENTS, 1.0);
    std::vector<std::thread> threads(NTHREADS);

    double result = 0.0;
    for (size_t t = 0; t < NTHREADS; ++t) {
        size_t const start = t * QOT;
        size_t const end = t == NTHREADS - 1 ? start + QOT + REM : start + QOT;
        threads[t] = std::thread([&]() {
            double partial_sum = 0.0;
            for (size_t i = start; i < end; ++i) {
                partial_sum += vector[i];
            }
            result += partial_sum;
        });
    }

    for (auto& t: threads) {
        t.join();
    }

    printf("result: %lf\n", result);
    return 0;
}

