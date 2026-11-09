//kernel de hls para probar el kernel de softmax con diferentes optimizaciones
// para dataflow fue necesario refactorizar el codigo en funciones que se comunican por fifos
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "hls_math.h"
#include "hls_half.h"
#include "hls_stream.h"

#ifdef USE_FLOAT32 // se usa esta logica para poder cambiar el tipo de dato son un flag
typedef float data_t; // este es de 32 bits
#else
typedef half data_t; // este es el de 16 bits
#endif

// esta funcion lee de la memoria global y duplica el dato en dos buffers para poder usar dataflow 
static void read_data(const data_t x[1000], hls::stream<data_t> &stream_max, hls::stream<data_t> &stream_exp, int n) {
for (int i = 0; i < n; i++) { // se copia el arreglo de x a los buffers locales
    #ifdef USE_PIPELINE
    #pragma HLS PIPELINE II=1
    #endif
    data_t v = x[i];
    stream_max.write(v);
    stream_exp.write(v);
  }
}

// aca se calcula el valor maximo del arreglo
static void find_max(hls::stream<data_t> &stream_max, data_t &m, int n) {
  data_t max_value = stream_max.read(); 
  for (int i = 1; i < n; i++) { // recorre el buffer buscando el valor maximo
    #ifdef USE_PIPELINE
    #pragma HLS PIPELINE II=1
    #endif
    data_t val = stream_max.read();
    if (val > max_value) {
      max_value = val;
    }
  }
  m = max_value;
}

// aca se calcula la exponencial de cada elemento y se acumula la suma
static void comp_exp(hls::stream<data_t> &stream_exp, data_t m, hls::stream<data_t> &stream_e_buf, data_t &s, int n) {
  data_t sum = 0.0; // variable para acumular la suma de las exponenciales
  for (int i = 0; i < n; i++) {
    #ifdef USE_PIPELINE
    #pragma HLS PIPELINE II=1
    #endif
    data_t val = stream_exp.read();
    data_t e = hls::exp(val - m);
    stream_e_buf.write(e);
    sum += e;
  }
  s = sum;
}

// aca se normalizan las exponenciales y se escriben en la memoria global
static void norm_exp(hls::stream<data_t> &stream_e_buf, data_t s, data_t y[1000], int n) {  
    for (int i = 0; i < n; i++) {
    #ifdef USE_PIPELINE
    #pragma HLS PIPELINE II=1
    #endif
    y[i] = stream_e_buf.read() / s;
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

  // declaracion de los streams locales para poder usar datadflow
  hls::stream<data_t> stream_max("stream_max");
  hls::stream<data_t> stream_exp("stream_exp");
  hls::stream<data_t> stream_e_buf("stream_e_buf");
  data_t m;
  data_t s;

#ifdef USE_PARTITION
//solo los arreglo de entrada se pueden particionar sin tener que volver a refactorizar todo el codigo xd
 #pragma HLS ARRAY_PARTITION variable=x cyclic factor=4
  #pragma HLS ARRAY_PARTITION variable=y cyclic factor=4
#endif

#ifdef USE_DATAFLOW
// estas fifos se agregan para que se pueda usar dataflow
  #pragma HLS STREAM variable=stream_max   depth=1000
  #pragma HLS STREAM variable=stream_exp   depth=1000
  #pragma HLS STREAM variable=stream_e_buf depth=1000
// se usa dataflow para que cada funcion pueda ejecutarse en paralelo
#pragma HLS DATAFLOW
#endif

  read_data(x, stream_max, stream_exp, n);
  find_max(stream_max, m, n);
  comp_exp(stream_exp, m, stream_e_buf, s, n);
  norm_exp(stream_e_buf, s, y, n);
}

}