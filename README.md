# Búsquedas locales: Hill Climbing, sus variantes y Recocido Simulado en la función esfera

Implementación en C++ de cuatro búsquedas locales para minimizar la función esfera con 10, 30, 50 y 100 dimensiones:

```
Minimizar f(x) = Σ xi²,  i = 1..D
-10 ≤ xi ≤ 10
Óptimo conocido: f(0, …, 0) = 0
```

**Autor:** Angel Gabriel Aguilar Hernandez · zs24013324@estudiantes.uv.mx
Licenciatura en Ingeniería de Software, Universidad Veracruzana

## Compilación y ejecución

Requiere un compilador compatible con C++17 (`clang++` o `g++`). Los algoritmos y las estadísticas están programados desde cero; solo se usa la biblioteca estándar.

```bash
clang++ -std=c++17 -O2 main.cpp -o local_search
./local_search
```

Opcionalmente recibe el número de hilos para la fase de ajuste (por omisión, todos los núcleos): `./local_search 8`. En un Apple M2 tarda unos 7 minutos.

Las figuras del reporte se generan con Python y matplotlib (solo para graficar):

```bash
python3 plot_results.py
```

## Algoritmos

Siguen los pseudocódigos de *Essentials of Metaheuristics* (S. Luke, 2.ª ed.):

| Algoritmo | Pseudocódigo | Regla |
|---|---|---|
| Hill Climbing | Algoritmo 4 | Un vecino por iteración; se acepta solo si mejora |
| Steepest Ascent | Algoritmo 5 | n vecinos; el mejor se acepta solo si mejora |
| Steepest Ascent with Replacement | Algoritmo 6 | n vecinos; siempre se pasa al mejor y se guarda *Best* |
| Simulated Annealing | Algoritmo 13 | Un vecino; se acepta un empeoramiento con probabilidad e^((f(S) − f(R)) / T) |

Los cuatro usan la convolución uniforme acotada (Algoritmo 8) como operador *Tweak*: cada componente se modifica con probabilidad p = k/D sumándole ruido uniforme en [−r, r]. El recocido enfría geométricamente desde T0 hasta 10⁻¹⁰ al agotar el presupuesto.

## Diseño experimental

| Elemento | Valor |
|---|---|
| Presupuesto | 10,000 · D evaluaciones (criterio del CEC 2005) |
| Paro anticipado | f(x) ≤ 10⁻⁸ (ejecución exitosa) |
| Ejecuciones | 30 por algoritmo y dimensión, semillas 1001 a 1030 |
| Ajuste de parámetros | Rejilla con 5 ejecuciones piloto por configuración, semillas 5001 a 5005 |
| Rejilla | k ∈ {1, 2, 5, 10}, r ∈ {0.001, 0.01, 0.1, 1}, n ∈ {2, 5, 10, 20, 50}, T0 ∈ {0.001, 0.01, 0.1, 1, 10} |
| Tiempo | Tiempo de CPU (`std::clock`), ejecuciones finales en secuencia |

La configuración elegida es la de más ejecuciones exitosas; en caso de empate, la de menor mediana de f (recortada en 10⁻⁸), y después la de menos evaluaciones. Todos los parámetros están definidos como constantes al inicio de `main.cpp`.

## Reproducibilidad

Se usa `std::mt19937`. En la ejecución *i* los cuatro algoritmos reciben la semilla `BASE_SEED + i`, por lo que parten de la misma solución inicial. Con el mismo compilador se obtienen exactamente los mismos valores de f y de evaluaciones; solo el tiempo varía entre corridas.

## Archivos

| Archivo | Contenido |
|---|---|
| `main.cpp` | Algoritmos, ajuste de parámetros y experimento |
| `plot_results.py` | Figuras de convergencia y de dispersión |
| `results/tuning.csv` | Las 1,024 configuraciones probadas y su desempeño |
| `results/parameters.csv` | Configuración elegida por algoritmo y dimensión |
| `results/runs.csv` | Resultado de cada una de las 480 ejecuciones |
| `results/summary.csv` | Mejor, peor, media, mediana, desviación estándar, tiempo, evaluaciones y éxitos |
| `results/convergence.csv` | Mejor f(x) en 100 puntos de control de cada ejecución |
| `figures/` | Figuras del reporte |
| `Busquedas_locales_funcion_esfera.pdf` | Reporte en formato LNCS (entregable) |
| `Busquedas_locales_funcion_esfera.docx` | Fuente del reporte, sobre la plantilla LNCS |
# busquedas-locales
