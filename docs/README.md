# Pong AI

 Este proyecto implementa una versión del clásico videojuego **Pong**, potenciada con inteligencia artificial para jugar de manera autónoma.

## Tabla de Contenidos

- [Descripción](#descripción)
- [Características](#características)
- [Instalación](#instalación)
- [Uso](#uso)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Contribuciones](#contribuciones)
- [Licencia](#licencia)
- [Contacto](#contacto)

---

## Descripción

Este proyecto utiliza **C++** para crear una simulación del clásico juego de Pong, con una integración de inteligencia artificial que permite a un jugador autónomo aprender y mejorar su desempeño. Es ideal para explorar conceptos de desarrollo de videojuegos e inteligencia artificial.

---

## Características

- 🎮 **Juego Clásico de Pong**: Revive el clásico Pong con una interfaz moderna.
- 🧠 **Inteligencia Artificial**: Implementación de un modelo de IA que aprende a jugar.
- 🚀 **Optimización en C++**: Código altamente optimizado para rendimiento.
- 🛠️ **Fácil Personalización**: Modifica las reglas del juego o ajusta los parámetros de la IA.

---

## Instalación

Sigue estos pasos para clonar y ejecutar el proyecto en tu máquina local:

1. Clona este repositorio:
   ```bash
   git clone https://github.com/stiffis/pong-ia.git
   cd pong-ia
   ```

2. Compila el proyecto (asegúrate de tener un compilador de C++ instalado):
   ```bash
   # This will build the training executable (pong_ai_train) and test executables.
   ./build.sh
   ```

3. Ejecuta el juego:
   ```bash
   # To run the training process:
   ./pong_ai_train
   # Test executables are built into the build/tests/ directory.
   # For example: ./build/tests/test_tensor_run
   ```

---

## Uso
1. Al iniciar el juego, puedes elegir entre:
   - Jugar contra la IA.
   - Ver cómo dos IAs compiten entre sí.
   (Nota: La funcionalidad de juego interactivo o IA vs IA aún no está implementada en `pong_ai_train`.
   El ejecutable `pong_ai_train` corre el proceso de entrenamiento de la IA.)

---

## Estructura del Proyecto

```txt
PongIA
├── build.sh
├── docs
│   ├── DOCS.md
│   └── README.md
├── include
│   └── utec
│       ├── agent
│       │   ├── EnvGym.cpp
│       │   ├── EnvGym.h
│       │   ├── PongAgent.cpp
│       │   ├── PongAgent.h
│       │   └── state.h
│       ├── algebra
│       │   ├── matmul.h
│       │   └── Tensor.h
│       └── nn
│           ├── activation.h
│           ├── dense.h
│           ├── layer.h
│           ├── loss.h
│           └── neural_network.h
├── LICENSE
├── src
├── temp
│   └── train.cpp
├── tests
│   ├── test_agent_env.cpp
│   ├── test_benchmark_matmul.cpp
│   ├── test_matmul.cpp
│   ├── test_neural_network.cpp
│   └── test_tensor.cpp
└── tree.txt

10 directories, 23 files
```

---

## Contribuciones

¡Las contribuciones son bienvenidas! Si deseas colaborar:

1. Realiza un fork del repositorio.
2. Crea un rama para tu nueva funcionalidad:
   ```git
   git checkout -b nueva-funcionalidad
   ```
3. Realiza tus cambios y súbelos.
   ```git
   git add .
   git commit -m "feat: add new feature"
   git push origin nueva-funcionalidad
   ```
4. Abre un **Pull Request**

---

## Licencia

Este proyecto está bajo la licencia **MIT**. Consulta el archivo [LICENSE](LICENSE) para más detalles.

## Contacto

- **Autor**: [stiffis](https://github.com/stiffis)
- Si tienes preguntas o sugerencias, no dudes en abrir un issue o contactarme directamente.

---
```cat
⠀ ／l
（ﾟ､ ｡ ７
⠀ l、ﾞ ~ヽ
  じしf_, )ノ ❤️
```
steps:
- make a main.cpp file
- update directory tree

