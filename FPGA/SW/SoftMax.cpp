//NOTA: este codigo no es el codigo que se ejecutara en la cpu de la kria, es solamente el codigo base  para entender el algoritmo
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#define NUM_ELEMENTS 1000 // default
#define DEFAULT_REPETITIONS 100000 // default


// inicializa el vector de entrada con valores entre -4.0 y 4.0
static void init_input(double *x, size_t n) {
  for (size_t i = 0; i < n; i++) {
    x[i] = ((double)(i % 97) / 10.0) - 4.0;
  }
}
// calcula softmax de un vector de entrada x y almacena el resultado en el vector de salida
static void softmax(const double *x, double *y, size_t n) {
  double max_value = x[0];

  for (size_t i = 0; i < n; i++) { // encuentra el valor maximo para estabilidad numerica (que no haya overflow)
    if (x[i] > max_value) {
      max_value = x[i];
    }
  }

  double sum = 0.0;

  for (size_t i = 0; i < n; i++) { // calcula la exponencial de cada elemento y acumula la suma
    y[i] = exp(x[i] - max_value);
    sum += y[i];
  }

  for (size_t i = 0; i < n; i++) { //normaliza la exponencial de cada elemento dividiendolo entre la suma de todas las exponenciales
    y[i] /= sum;
  }
}

static double sum_array(const double *x, size_t n) { // suma todos los elementos del array y devuelve el resultado
  double sum = 0.0;

  for (size_t i = 0; i < n; i++) {
    sum += x[i];
  }

  return sum;
}

int main(int argc, char **argv) {
  int repetitions = DEFAULT_REPETITIONS;
  // se declara un array de entrada y un array de salida con el tamaño definido por NUM_ELEMENTS
  double input[NUM_ELEMENTS];
  double output[NUM_ELEMENTS];

  init_input(input, NUM_ELEMENTS); // se llena el array de entrada con valores entre -4.0 y 4.0

  auto start = std::chrono::high_resolution_clock::now(); // se inicia el cronometro para medir el tiempo de ejecucion

  for (int i = 0; i < repetitions; i++) { // aca se calcula el softmax de la entrada y se almacena en la salida, repitiendo el proceso 'repetitions' veces
    softmax(input, output, NUM_ELEMENTS); // aca no se por qué se repite tantas veces, pero supongo que es para hacer benchmarking, esto talvez se pueda omitir en los kernels
  }

  auto end = std::chrono::high_resolution_clock::now(); // se detiene el cronometro

  double total = sum_array(output, NUM_ELEMENTS); // calcular el total de todos los elementos del softmax, debe ser 1

  printf("Elementos: %d\n", NUM_ELEMENTS);
  printf("Repeticiones: %d\n", repetitions);
  printf("Suma softmax: %.17f\n", total);
  printf("Tiempo: %.6f segundos\n", std::chrono::duration<double>(end - start).count());
  printf("Primeros valores: %.12f %.12f %.12f %.12f %.12f\n", output[0],
         output[1], output[2], output[3], output[4]);
  return EXIT_SUCCESS;
}