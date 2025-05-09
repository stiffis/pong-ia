
# Plan de Proyecto: Pong AI en C++ 20 (alineado al documento CS2013‑Programación III 2025 – UTEC)

## 1. Propósito

Desarrollar un **agente de IA competitivo** capaz de ganar en el fixture de la asignatura jugando a una versión simplificada de Pong, respetando los cinco *Epics* exigidos en el PDF oficial.

## 2. Epics y entregables

| Epic                                    | Objetivo                                                                                        | Entregable clave                                                            | Criterio de aceptación                                                |
| --------------------------------------- | ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| **1. Biblioteca Genérica de Álgebra**   | Implementar `utec::algebra::Tensor<T,Rank>` con operadores básicos y tests.                     | `include/utec/algebra/tensor.h` + `tests/test_tensor.cpp`                   | Todos los casos nº 1‑7 del anexo pasan en < 2 ms.                     |
| **2. Red Neuronal Full**                | Crear framework de capas densas + ReLU + MSELoss sobre la biblioteca de álgebra.                | `include/utec/nn/*.h` + `tests/test_neural_network.cpp`                     | Entrena XOR con pérdida < 0.1 en < 1 s.                               |
| **3. PongAgent & EnvGym**               | Diseñar entorno minimalista y agente que use la red entrenada para decidir la acción {-1,0,+1}. | `include/utec/agent/{EnvGym,PongAgent}.h/.cpp` + `tests/test_agent_env.cpp` | Agente devuelve ≥ 70 % de pelotas tras 50 000 pasos de entrenamiento. |
| **4. Paralelismo y CUDA (opcional)**    | Acelerar inferencias con `std::jthread` y/o kernels CUDA.                                       | `src/thread_pool.*`                                                         | > 50 000 steps/s en portátil i5 (CPU) o > 5 × speed‑up con GPU.       |
| **5. Entrenamiento, Validación y Docs** | Pipeline de entrenamiento + README + BIBLIOGRAFIA.md.                                           | `train.cpp`, `README.md`                                                    | Agente devuelve ≥ 95 % pelotas y obtiene > nivel 3 en fixture mock.   |

## 3. Arquitectura del código

```
pong_ai/
├─ include/
│  ├─ utec/algebra/tensor.h          # Epic 1
│  ├─ utec/nn/{layer,dense,activation,loss,neural_network}.h  # Epic 2
│  └─ utec/agent/{EnvGym,PongAgent}.h                         # Epic 3
├─ src/      # Implementaciones *.cpp, ThreadPool (Epic 4)
├─ tests/    # Catch2 test_*.cpp para Gradescope
├─ train.cpp # Entrenamiento y serialización (Epic 5)
└─ docs/README.md, BIBLIOGRAFIA.md
```

## 4. Simulación **simple pero eficaz** (EnvGym)

* **Estado**: `{ball_x, ball_y, ball_vx, ball_vy, paddle_y}` ∈ \[0,1].
* **Acción**: `-1` (subir), `0` (quieto), `+1` (bajar).
* **Recompensa**: `+1` devuelve bola, `-1` falla, `0` en otro paso.
  *Tip:* Añadir castigo suave `−|ball_y − paddle_y|·λ` acelera la convergencia.
* **Paso** (`step()`): Integración simple, colisiones elásticas y condición `done` cuando la bola sale por la izquierda.
* **Velocidad**: Física sin *std::sin/cos*; solo sumas/restas para alcanzar > 100 k steps/s si se compila con `-O3`.

## 5. Agente y entrenamiento

* **Modelo base**: MLP 3‑4‑1 (entrada 5, capa oculta 32 unidades, salida Q‑values de 3 acciones).
* **Algoritmo**: *Hill‑Climbing* + pequeña exploración ε‑greedy (recomendado por PDF) — simple, sin backprop dinámico en gameplay.
* **Serialización**: Guardar pesos en `model.dat` usando fstream binario.

## 6. Iteraciones (semanas)

| Semana | Entregables                           | Metas de rendimiento |
| ------ | ------------------------------------- | -------------------- |
| 1      | Tensor\<T,Rank> + tests               | 100 % casos PDF      |
| 2      | NeuralNetwork mínima + XOR            | XOR listo < 1 s      |
| 3      | EnvGym + PongAgent prototipo          | ≥ 70 % pelotas       |
| 4      | Paralelismo CPU / ajustar recompensas | ≥ 90 %               |
| 5      | Pulido, docs, fixture mock            | ≥ 95 %, reproducible |

## 7. Herramientas

* **C++ 20** (g++ 14 +, clang 16 +).
* **CMake 3.20**.
* **Catch2 v3** para tests.
* **fmtlib** para *logging* (ligero).
* **CUDA 12** opcional.

## 8. Próximos pasos inmediatos

1. Crear repositorio Git y plantilla CMake.
2. Implementar `Tensor<T,Rank>` + 7 tests PDF (Epic 1).
3. Revisar rendimiento y corregir densidades de bucle.
4. Volver a esta hoja para aprobar Sprint 1 antes de avanzar.

---

**Mantendremos todo “simple, eficiente y eficaz”, enfocándonos en pasar los tests y en entrenar lo justo para superar a otras IA en el fixture.**
