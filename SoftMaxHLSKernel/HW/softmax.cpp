#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "hls_math.h"

extern "C" {

void softmax(int32_t *x, int32_t *y, int n, double sum_total) {
#pragma HLS INTERFACE m_axi port = x offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = y offset = slave bundle = gmem1

#pragma HLS INTERFACE s_axilite register port = n // el n es el tamaño del vector de entrada y salida
#pragma HLS INTERFACE s_axilite register port = sum_total // el sum_total es la suma de todos los elementos del vector de salida, debe ser 1
 
// aca va el algoritmo de softmax de HLS
double max_value = x[0];

  for (size_t i = 0; i < n; i++) { // encuentra el valor maximo para estabilidad numerica (que no haya overflow)
    if (x[i] > max_value) {
      max_value = x[i];
    }
  }

  double sum = 0.0; // variable para acumular la suma de las exponenciales

  for (size_t i = 0; i < n; i++) { // calcula la exponencial de cada elemento y acumula la suma
    y[i] = hls::exp(x[i] - max_value);
    sum += y[i];
  }

  for (size_t i = 0; i < n; i++) { //normaliza la exponencial de cada elemento dividiendolo entre la suma de todas las exponenciales
    y[i] /= sum;
  }

  // esto es solo para reportar el total de todos los elementos del array de salida, debe ser 1
  for (size_t i = 0; i < n; i++) {
    sum_total += x[i];
  }

}
}
