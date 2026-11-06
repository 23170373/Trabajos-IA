# Tarea #6. Dataset propio de rostros

**Materia:** Inteligencia Artificial  
**Profesor:** José Mario Ríos Félix  
**Instituto Tecnológico de Culiacán**

Programa desarrollado en Python utilizando OpenCV y el algoritmo Haar Cascade para detectar, recortar y guardar imágenes faciales. El objetivo es crear un dataset de **900 fotografías**, con 100 imágenes de cada integrante del equipo y de cinco famosos.

Posteriormente, este dataset se utilizará para entrenar un modelo de reconocimiento facial.

## Integrantes del equipo

- Kitzia Marisol Crespo Hernández
- Emir Gabino Pérez Villarreal
- Oscar Antonio Sánchez Lara
- Jonathan Daniel Valencia

## Famosos seleccionados

- Robert Downey Jr.
- Emma Watson
- Elizabeth Olsen
- Alexandra Daddario
- Dwayne Johnson

Las fotografías se obtuvieron del **105 Classes Pins Dataset** de Kaggle, seleccionando famosos con suficientes imágenes para realizar el preprocesamiento.

## Instalación y ejecución

Instalar las dependencias desde la terminal:

```bash
pip install -r requirements.txt
```
Ejecutar el programa:

```bash
python dataset_rostros.py
```

Al iniciar, se muestra el siguiente menú:

```text
DATASET DE ROSTROS
1. Capturar rostros con la cámara
2. Recortar las fotos de los famosos
3. Salir
```

## Funcionamiento

El programa cuenta con dos opciones principales:

**1. Capturar rostros con la cámara**

Permite tomar hasta 100 fotografías por integrante, detectando y recortando automáticamente el rostro.

| Tecla | Función |
| --- | --- |
| ESPACIO | Iniciar o pausar |
| R | Girar la imagen 90° |
| Q | Finalizar |

**2. Procesar fotografías de famosos**

Detecta y recorta los rostros de las imágenes descargadas, ubicadas en `fotos_famosos_sin_recortar/`.

Si no se completan las 100 imágenes, el programa realiza búsquedas adicionales con otros clasificadores Haar Cascade y revisa los recortes obtenidos.

## Estructura de carpetas

```text
Tarea 6. Dataset propio/
├── dataset_rostros.py
├── requirements.txt
├── README.md
├── haarcascade_frontalface_default.xml
├── fotos_famosos_sin_recortar/
└── dataset/
    ├── integrantes_equipo/
    │   ├── Kitzia_Crespo/
    │   ├── Emir_Perez/
    │   ├── Oscar_Sanchez/
    │   └── Jonathan_Valencia/
    └── famosos/
        ├── Robert_Downey_Jr/
        ├── Emma_Watson/
        ├── Elizabeth_Olsen/
        ├── Alexandra_Daddario/
        └── Dwayne_Johnson/
```

La carpeta `fotos_famosos_sin_recortar/` no se incluye en el repositorio debido al tamaño de las fotografías originales descargadas de Kaggle. Solo se incorporan las imágenes procesadas del dataset.

## Recomendaciones

Durante la captura, se recomienda utilizar buena iluminación y variar ligeramente la posición del rostro, las expresiones y el uso de lentes cuando corresponda, y al finalizar, se deben revisar las imágenes para comprobar que los rostros estén correctamente recortados.

## Parámetros de configuración

Los siguientes valores se pueden modificar al inicio de `dataset_rostros.py`:

| Parámetro | Valor | Descripción |
| --- | --- | --- |
| `CARPETA_DE_FOTOS_FAMOSOS` | `""` | Ruta alternativa de las fotografías originales |
| `IMAGENES_POR_PERSONA` | `100` | Cantidad máxima de imágenes por persona |
| `TAMANO_RECORTE` | `160` | Resolución de salida de 160 × 160 píxeles |
| `MARGEN` | `0.0` | Espacio adicional alrededor del rostro |
| `NITIDEZ_MINIMA` | `25.0` | Umbral para filtrar imágenes borrosas |
| `GUARDAR_EN_GRIS` | `False` | Permite guardar imágenes en escala de grises |
| `MODO_ESPEJO` | `True` | Invierte horizontalmente la vista de la cámara |
| `ROTACION` | `0` | Orientación de la cámara en grados |
| `INDICE_CAMARA` | `0` | Cámara que se intenta utilizar inicialmente |

## Notas técnicas

- Si se detectan varios rostros en una imagen, se selecciona el que ocupa mayor espacio.
- Se incluye `haarcascade_frontalface_default.xml` como respaldo del clasificador de OpenCV.
- La cámara puede mostrarse en espejo, pero las fotografías se guardan sin invertir horizontalmente.