// kernel de hls para probar el kernel de softmax con diferentes optimizaciones
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "hls_math.h"
#include "hls_half.h"

#ifdef USE_FLOAT32 // se usa esta logica para poder cambiar el tipo de dato son un flag
typedef float data_t; // este es de 32 bits
#else
typedef half data_t; // este es el de 16 bits
#endif

// aca la idea de esta logica es que si se usa la directiva de particion, que los bucles salten de 4 en 4 (4 operacion en paralelo) en vez de 1 en 1
#ifdef USE_PARTITION
#define UF 4
#else
#define UF 1 // aca si no se usa particion, se deja en 1 para tener una version secuencial
#endif

// esta funcion lee de la memoria global y duplica el dato en dos buffers (cada buffer tiene un solo lector, necesario para dataflow)
static void read_data(const data_t x[1000], data_t buf_max[1000], data_t buf_exp[1000], int n) {
  for (int i = 0; i < n; i++) { // se copia el arreglo de x a los buffers locales
#ifdef USE_PIPELINE
#pragma HLS PIPELINE II=1
#endif
    data_t v = x[i];
    buf_max[i] = v;
    buf_exp[i] = v;
  }
}

// aca se calcula el valor maximo del arreglo, con un maximo parcial por carril para romper la dependencia entre iteraciones
static void find_max(const data_t buf_max[1000], data_t &m, int n) {
  data_t part[UF];
#pragma HLS ARRAY_PARTITION variable=part complete dim=1
  for (int k = 0; k < UF; k++) {
#pragma HLS UNROLL
    part[k] = buf_max[0];
  }

  for (int i = 0; i < n; i += UF) { // recorre el buffer buscando el valor maximo
#ifdef USE_PIPELINE
#pragma HLS PIPELINE II=1
#endif
    for (int k = 0; k < UF; k++) {
#pragma HLS UNROLL
      if (i + k < n && buf_max[i + k] > part[k]) { // la condicion evita leer fuera de n cuando n no es multiplo de UF
        part[k] = buf_max[i + k];
      }
    }
  }

  data_t max_value = part[0];
  for (int k = 1; k < UF; k++) { // reduccion final de los maximos parciales
#pragma HLS UNROLL
    if (part[k] > max_value) {
      max_value = part[k];
    }
  }
  m = max_value;
}

// aca se calcula la exponencial de cada elemento y se acumula la suma (una suma parcial por carril)
static void comp_exp(const data_t buf_exp[1000], data_t m, data_t e_buf[1000], data_t &s, int n) {
  data_t part[UF]; // sumas parciales para acumular las exponenciales
#pragma HLS ARRAY_PARTITION variable=part complete dim=1
  for (int k = 0; k < UF; k++) {
#pragma HLS UNROLL
    part[k] = 0;
  }

  for (int i = 0; i < n; i += UF) {
#ifdef USE_PIPELINE
#pragma HLS PIPELINE II=1
#endif
    for (int k = 0; k < UF; k++) {
#pragma HLS UNROLL
      if (i + k < n) {
        data_t e = hls::exp(buf_exp[i + k] - m);
        e_buf[i + k] = e;
        part[k] += e;
      }
    }
  }

  data_t sum = 0;
  for (int k = 0; k < UF; k++) { // reduccion final de las sumas parciales
#pragma HLS UNROLL
    sum += part[k];
  }
  s = sum;
}

// aca se normalizan las exponenciales y se escriben en la memoria global
static void norm_exp(const data_t e_buf[1000], data_t s, data_t y[1000], int n) {
  for (int i = 0; i < n; i += UF) {
#ifdef USE_PIPELINE
#pragma HLS PIPELINE II=1
#endif
    for (int k = 0; k < UF; k++) {
#pragma HLS UNROLL
      if (i + k < n) {
        y[i + k] = e_buf[i + k] / s;
      }
    }
  }
}

extern "C" {

void softmax_hls(const data_t x [1000], data_t y [1000], int n) {
// registro de control para el axi
#pragma HLS INTERFACE m_axi     port=x offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi     port=y offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=x      bundle=control
#pragma HLS INTERFACE s_axilite port=y      bundle=control
#pragma HLS INTERFACE s_axilite port=n      bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  // buffers locales entre etapas (un escritor y un lector cada uno)
  data_t buf_max[1000];
  data_t buf_exp[1000];
  data_t e_buf[1000];
  data_t m;
  data_t s;

#ifdef USE_PARTITION
// los puertos m_axi no se pueden particionar, por eso se particionan los buffers locales
#pragma HLS ARRAY_PARTITION variable=buf_max cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=buf_exp cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=e_buf   cyclic factor=4
#endif

#ifdef USE_DATAFLOW

#pragma HLS DATAFLOW
#endif

  read_data(x, buf_max, buf_exp, n);
  find_max(buf_max, m, n);
  comp_exp(buf_exp, m, e_buf, s, n);
  norm_exp(e_buf, s, y, n);
}

}