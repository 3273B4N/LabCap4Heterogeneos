// testbench para para probar el kernel de softmax 
#include <cstdio>
#include <cmath>
#include "hls_half.h"

#ifdef USE_FLOAT32
typedef float data_t;
#else
typedef half data_t;
#endif

extern "C" void softmax_hls(const data_t x[1000], data_t y[1000], int n);

int main() {
#ifdef USE_FLOAT32
  const double tol = 1e-5;
#else
  const double tol = 1e-2;
#endif
  const int sizes[4] = {10, 100, 500, 1000};
  static data_t x[1000], y[1000];
  static double xr[1000], ref[1000];
  int fallos = 0;

  for (int k = 0; k < 4; k++) {
    int n = sizes[k];
    for (int i = 0; i < n; i++) {
      x[i] = data_t((i % 97) / 10.0 - 4.8);
      xr[i] = (double)(float)x[i];
    }
    double m = xr[0], s = 0.0;
    for (int i = 1; i < n; i++) if (xr[i] > m) m = xr[i];
    for (int i = 0; i < n; i++) { ref[i] = exp(xr[i] - m); s += ref[i]; }
    for (int i = 0; i < n; i++) ref[i] /= s;

    softmax_hls(x, y, n);

    double err = 0.0, suma = 0.0;
    for (int i = 0; i < n; i++) {
      double d = fabs((double)(float)y[i] - ref[i]);
      if (d > err) err = d;
      suma += (double)(float)y[i];
    }
    printf("N=%d  error max=%.3e  suma=%.5f\n", n, err, suma);
    if (err > tol) fallos++;
  }
  printf(fallos ? "RESULTADO: FALLO\n" : "RESULTADO: OK\n");
  return fallos ? 1 : 0;
}