"""
Clasificación de rostros en tiempo real mediante redes neuronales convolucionales (CNN)

Materia: Inteligencia Artificial
Docente: José Mario Ríos Félix
Horario: 6:00 a 7:00 p. m.
Instituto Tecnológico de Culiacán

Integrantes:
    - Kitzia Marisol Crespo Hernández. C23171148
    - Emir Gabino Pérez Villarreal. 23170278
    - Oscar Antonio Sánchez Lara. 23170334
    - Jonathan Daniel Valencia. 23170373

Descripción:
    Este programa realiza reconocimiento facial en tiempo real mediante una
    red neuronal convolucional (CNN) entrenada con el dataset de rostros de la tarea #6,
    que incluye a los cuatro integrantes del equipo y cinco personas famosas.
    Durante su ejecución, captura el video de la cámara web, localiza los rostros
    con un clasificador Haar Cascade y clasifica cada rostro detectado con el modelo entrenado.

    Sobre cada rostro se dibuja una caja delimitadora con el nombre de la
    persona identificada y el porcentaje de confianza. Cuando la confianza
    no alcanza el umbral mínimo, el rostro se muestra como desconocido.

    El modelo se carga desde el archivo modelo_rostros.keras, generado por la
    notebook entrenar-cnn-rostros.ipynb, sin necesidad de repetir el
    entrenamiento.

Controles:
    ESPACIO     pausa y reanuda el video
    E           activa o desactiva la vista en espejo
    G           guarda una captura de la ventana
    Q o ESC     cierra el programa

Ejecución:
    python reconocer-rostros.py
"""

import json
import logging
import os
import shutil
import sys
import tempfile
import time
from collections import deque

# Reduce los mensajes de avisos de TensorFlow en la consola
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Desactiva las optimizaciones oneDNN para evitar sus avisos al iniciar TensorFlow
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# Oculta los mensajes informativos y advertencias del registro de TensorFlow
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import cv2
import numpy as np


# Obtiene la ubicación del programa para localizar el modelo y las clases
CARPETA_BASE = os.path.dirname(os.path.abspath(__file__))

# Archivo con el modelo entrenado por la notebook
ARCHIVO_MODELO = os.path.join(CARPETA_BASE, "modelo_rostros.keras")

# Archivo con los nombres de las clases en el mismo orden que usó el modelo
ARCHIVO_CLASES = os.path.join(CARPETA_BASE, "clases.json")

# Archivo del clasificador Haar Cascade utilizado para la detección de rostros
ARCHIVO_CASCADA = os.path.join(CARPETA_BASE, "haarcascade_frontalface_default.xml")

# Medida a la que se redimensiona cada rostro si el modelo no la indica
LADO_ENTRADA = 128

# Umbral mínimo de confianza para identificar un rostro (60 %)
CONFIANZA_MINIMA = 0.60

# Cantidad de cuadros que se promedian para estabilizar cada predicción
CUADROS_SUAVIZADO = 8

# Margen adicional alrededor del rostro, igual al utilizado en la preparación del dataset
MARGEN = 0.0

# Cámara que se intenta utilizar inicialmente
INDICE_CAMARA = 0

# Resolución que se le solicita a la cámara
ANCHO_CAMARA = 800
ALTO_CAMARA = 600

# Configuración de la vista en espejo y los parámetros de detección de rostros
MODO_ESPEJO = True
ESCALA_DETECCION = 1.12
VECINOS_MINIMOS = 6
LADO_MINIMO_ROSTRO = 66

# Alto de las franjas de información de la ventana
ALTO_ENCABEZADO = 64
ALTO_PIE = 44

# Colores de la interfaz definidos en formato RGB
# Se usa la misma paleta oscura del programa que armó el dataset en la tarea 6
COLOR_FONDO = (16, 18, 23)
COLOR_TEXTO = (209, 213, 219)
COLOR_SUAVE = (150, 157, 170)
COLOR_BORDE = (48, 53, 66)

# Configuración visual de la interfaz: colores, indicador de estado y subtítulo
COLOR_TITULO = (250, 204, 21)
COLOR_CINTA = (16, 18, 23)
COLOR_ACIERTO = (74, 222, 128)
COLOR_DUDA = (245, 158, 11)
RADIO_PUNTO = 4
SUBTITULO = "Red neuronal convolucional"

# Pillow permite escribir con acentos y con una tipografía del sistema
try:
    from PIL import Image, ImageDraw, ImageFont
    HAY_PILLOW = True
except ImportError:
    HAY_PILLOW = False

RUTAS_DE_FUENTE = [
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
RUTAS_DE_FUENTE_NEGRITA = [
    r"C:\Windows\Fonts\segoeuib.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
FUENTES_CARGADAS = {}


def a_bgr(color):
    """Cambia un color de rojo, verde, azul al orden que usa OpenCV."""
    return (color[2], color[1], color[0])


def obtener_fuente(tamano, negrita=False):
    """Carga una fuente del sistema y la guarda para no volver a leerla."""
    if not HAY_PILLOW:
        return None
    clave = (tamano, negrita)
    if clave not in FUENTES_CARGADAS:
        FUENTES_CARGADAS[clave] = None
        if negrita:
            rutas = RUTAS_DE_FUENTE_NEGRITA + RUTAS_DE_FUENTE
        else:
            rutas = RUTAS_DE_FUENTE
        for ruta in rutas:
            if os.path.isfile(ruta):
                try:
                    FUENTES_CARGADAS[clave] = ImageFont.truetype(ruta, tamano)
                    break
                except OSError:
                    continue
    return FUENTES_CARGADAS[clave]


def quitar_acentos(texto):
    """Elimina los acentos del texto para que OpenCV pueda mostrarlo correctamente."""
    import unicodedata
    normal = unicodedata.normalize("NFKD", texto.replace("·", "-"))
    return "".join(c for c in normal if not unicodedata.combining(c))


def escribir(vista, textos):
    """Muestra los textos con el formato definido, utilizando Pillow u OpenCV."""
    if obtener_fuente(14) is None:
        for texto, posicion, tamano, negrita, color in textos:
            escala = tamano / 32.0
            grosor = 2 if negrita else 1
            plano = quitar_acentos(texto)
            alto = cv2.getTextSize(plano, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)[0][1]
            cv2.putText(vista, plano, (posicion[0], posicion[1] + alto),
                        cv2.FONT_HERSHEY_SIMPLEX, escala, a_bgr(color), grosor, cv2.LINE_AA)
        return vista

    lamina = Image.fromarray(cv2.cvtColor(vista, cv2.COLOR_BGR2RGB))
    pincel = ImageDraw.Draw(lamina)
    for texto, posicion, tamano, negrita, color in textos:
        fuente = obtener_fuente(tamano, negrita)
        pincel.text(posicion, texto, font=fuente, fill=color)
    vista[:] = cv2.cvtColor(np.array(lamina), cv2.COLOR_RGB2BGR)
    return vista


def medir_texto(texto, tamano, negrita=False):
    """Calcula el ancho y el alto del texto según la fuente indicada."""
    fuente = obtener_fuente(tamano, negrita)
    if fuente is None:
        escala = tamano / 32.0
        grosor = 2 if negrita else 1
        (ancho, alto), _ = cv2.getTextSize(quitar_acentos(texto),
                                           cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)
        return ancho, alto
    izquierda, arriba, derecha, abajo = fuente.getbbox(texto)
    return derecha - izquierda, abajo - arriba


def medio_del_texto(texto, tamano, negrita=False):
    """Calcula el centro vertical del texto para alinear el punto indicador."""
    fuente = obtener_fuente(tamano, negrita)
    if fuente is None:
        return medir_texto(texto, tamano, negrita)[1] // 2
    izquierda, arriba, derecha, abajo = fuente.getbbox(texto)
    return (arriba + abajo) // 2


def ancho_fila(piezas, tamano):
    """Calcula el ancho total de los textos de una misma fila."""
    return sum(medir_texto(texto, tamano, negrita)[0] for texto, negrita, _ in piezas)


def textos_en_fila(piezas, x, y, tamano):
    """Organiza el texto en una misma línea y calcula sus posiciones."""
    textos = []
    avance = x
    for texto, negrita, color in piezas:
        textos.append((texto, (avance, y), tamano, negrita, color))
        avance += medir_texto(texto, tamano, negrita)[0]
    return textos


def cargar_modelo():
    """Carga la red neuronal convolucional entrenada por la notebook."""
    if not os.path.isfile(ARCHIVO_MODELO):
        print("No se encontró el archivo " + os.path.basename(ARCHIVO_MODELO))
        print("Primero hay que ejecutar la notebook entrenar-cnn-rostros.ipynb")
        print("y guarda ese archivo junto a este programa")
        sys.exit(1)

    print("Cargando el modelo, esto tarda unos segundos")
    from tensorflow import keras
    modelo = keras.models.load_model(ARCHIVO_MODELO)
    return modelo


def cargar_clases(cantidad_esperada):
    """
    Carga los nombres de las clases desde el archivo clases.json.
    Si el archivo no existe, se utilizan etiquetas genéricas para
    continuar con la ejecución del programa.
    """
    if os.path.isfile(ARCHIVO_CLASES):
        with open(ARCHIVO_CLASES, encoding="utf-8") as archivo:
            datos = json.load(archivo)
        nombres = datos["clases"] if isinstance(datos, dict) else datos
        if len(nombres) == cantidad_esperada:
            return list(nombres)
        print("El archivo de clases tiene " + str(len(nombres)) +
              " nombres y el modelo espera " + str(cantidad_esperada))

    print("No se encontró el archivo " + os.path.basename(ARCHIVO_CLASES))
    return ["Clase " + str(i + 1) for i in range(cantidad_esperada)]


def cargar_detector():
    """
    Carga el clasificador Haar Cascade incluido en OpenCV.
    Si no está disponible, se utiliza el archivo local desde una
    carpeta temporal para evitar problemas con rutas que contienen acentos.
    """
    carpeta_opencv = getattr(cv2.data, "haarcascades", "")
    if carpeta_opencv:
        ruta = os.path.join(carpeta_opencv, "haarcascade_frontalface_default.xml")
        if os.path.isfile(ruta):
            cascada = cv2.CascadeClassifier(ruta)
            if not cascada.empty():
                return cascada

    if os.path.isfile(ARCHIVO_CASCADA):
        copia = os.path.join(tempfile.gettempdir(), "haarcascade_frontalface_default.xml")
        try:
            shutil.copyfile(ARCHIVO_CASCADA, copia)
            cascada = cv2.CascadeClassifier(copia)
            if not cascada.empty():
                return cascada
        except OSError:
            pass

    print("No se pudo cargar el clasificador de rostros")
    sys.exit(1)


def abrir_camara():
    """
    Abre la cámara web, configura su resolución y prueba distintos
    índices para localizar una cámara disponible.
    """
    for indice in [INDICE_CAMARA, 0, 1, 2]:
        camara = cv2.VideoCapture(indice, cv2.CAP_DSHOW)
        if camara.isOpened():
            camara.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO_CAMARA)
            camara.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO_CAMARA)
            logrado, _ = camara.read()
            if logrado:
                print("Cámara abierta en el índice " + str(indice))
                return camara
        camara.release()

    print("No se pudo abrir la cámara, hay que revisar que no la esté usando otro programa")
    sys.exit(1)


def buscar_rostros(imagen, cascada):
    """
    Detecta los rostros de la imagen y los ordena de mayor a menor tamaño.
    Se aplica una ecualización del histograma, es decir, se ajusta el
    contraste de la imagen para facilitar la detección de rostros
    ante diferentes condiciones de iluminación.
    """
    gris = cv2.equalizeHist(cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY))
    cajas = cascada.detectMultiScale(gris, scaleFactor=ESCALA_DETECCION,
                                     minNeighbors=VECINOS_MINIMOS,
                                     minSize=(LADO_MINIMO_ROSTRO, LADO_MINIMO_ROSTRO))
    ordenadas = sorted(cajas, key=lambda caja: caja[2] * caja[3], reverse=True)
    return [tuple(int(valor) for valor in caja) for caja in ordenadas]


def recortar_rostro(imagen, caja, lado_salida):
    """
    Recorta el rostro en forma cuadrada y lo ajusta al tamaño de entrada
    de la red, con el mismo procedimiento utilizado para preparar el dataset.
    """
    x, y, ancho, alto = caja
    alto_imagen, ancho_imagen = imagen.shape[:2]

    lado = int(max(ancho, alto) * (1 + 2 * MARGEN))
    lado = min(lado, alto_imagen, ancho_imagen)

    centro_x = x + ancho // 2
    centro_y = y + alto // 2
    x1 = max(0, min(centro_x - lado // 2, ancho_imagen - lado))
    y1 = max(0, min(centro_y - lado // 2, alto_imagen - lado))

    recorte = imagen[y1:y1 + lado, x1:x1 + lado]
    interpolacion = cv2.INTER_AREA if lado >= lado_salida else cv2.INTER_LINEAR
    recorte = cv2.resize(recorte, (lado_salida, lado_salida), interpolation=interpolacion)
    return recorte, (x1, y1, lado, lado)


def preparar_para_la_red(recortes):
    """
    Organiza los rostros recortados en el arreglo que recibe la red.

    Cada rostro se convierte del formato BGR (azul, verde, rojo) al formato
    RGB (rojo, verde, azul), utilizado durante el entrenamiento. Los valores
    de los píxeles se mantienen entre 0 y 255, ya que el propio modelo
    se encarga de normalizarlos en su primera capa.
    """
    lote = [cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB) for recorte in recortes]
    return np.array(lote, dtype="float32")

def nombre_legible(clase):
    """Cambia los guiones bajos del nombre de la clase por espacios."""
    return clase.replace("_", " ")


class Seguimiento:
    """
    Guarda las últimas predicciones de un rostro para promediarlas.

    Promediar varios fotogramas evita cambios bruscos en la etiqueta
    cuando la persona se mueve o cambia momentáneamente la iluminación.
    """

    def __init__(self, centro, probabilidades):
        self.centro = centro
        self.historial = deque(maxlen=CUADROS_SUAVIZADO)
        self.historial.append(probabilidades)
        self.cuadros_sin_ver = 0

    def actualizar(self, centro, probabilidades):
        """Agrega una predicción nueva y mueve el centro del seguimiento."""
        self.centro = centro
        self.historial.append(probabilidades)
        self.cuadros_sin_ver = 0

    def promedio(self):
        """Devuelve el promedio de las probabilidades guardadas."""
        return np.mean(np.asarray(self.historial), axis=0)


def emparejar(seguimientos, centro, lado):
    """
    Identifica el seguimiento que corresponde a un rostro según su posición.

    Se considera el mismo rostro si su centro se desplaza menos de la
    mitad del lado de la caja delimitadora entre dos fotogramas consecutivos.
    """
    limite = lado * 0.5
    mejor = None
    menor = limite
    for seguimiento in seguimientos:
        distancia = float(np.hypot(seguimiento.centro[0] - centro[0],
                                   seguimiento.centro[1] - centro[1]))
        if distancia < menor:
            menor = distancia
            mejor = seguimiento
    return mejor


def dibujar_rostro(vista, caja, nombre, confianza, aceptado):
    """
    Dibuja la caja delimitadora del rostro con su etiqueta y el porcentaje
    de confianza.

    La caja se muestra en verde cuando la confianza alcanza el umbral
    mínimo y en ámbar cuando el rostro se clasifica como desconocido.
    """
    x, y, ancho, alto = caja
    color = COLOR_ACIERTO if aceptado else COLOR_DUDA
    cv2.rectangle(vista, (x, y), (x + ancho, y + alto), a_bgr(color), 3)

    piezas = [(nombre, True, COLOR_CINTA),
              ("   " + ("%.0f" % (confianza * 100)) + " %", False, COLOR_CINTA)]
    ancho_texto = ancho_fila(piezas, 19)
    alto_texto = medir_texto(nombre, 19, True)[1]
    alto_cinta = alto_texto + 16

    arriba = y - alto_cinta
    if arriba < 0:
        arriba = y + alto
    arriba = min(arriba, vista.shape[0] - alto_cinta)

    ancho_cinta = max(ancho, ancho_texto + 24)
    izquierda = min(x, vista.shape[1] - ancho_cinta)
    izquierda = max(0, izquierda)

    cv2.rectangle(vista, (izquierda, arriba),
                  (izquierda + ancho_cinta, arriba + alto_cinta), a_bgr(color), -1)

    # El texto se escribe solo sobre la franja de la cinta y no sobre toda la
    # imagen, porque convertirla completa para dibujar letras es muy lento
    franja = vista[arriba:arriba + alto_cinta]
    escribir(franja, textos_en_fila(piezas, izquierda + 12,
                                    (alto_cinta - alto_texto) // 2, 19))
    return vista


MARCOS_LISTOS = {}


def marco_base(ancho, alto):
    """
    Genera el fondo de la ventana con los elementos fijos, como
    las líneas separadoras, el título y la barra de controles.

    Se guarda el resultado para evitar dibujar estos elementos
    en cada fotograma y reducir el costo computacional.
    """
    clave = (ancho, alto)
    if clave in MARCOS_LISTOS:
        return MARCOS_LISTOS[clave]

    lienzo = np.zeros((alto + ALTO_ENCABEZADO + ALTO_PIE, ancho, 3), dtype=np.uint8)
    lienzo[:] = a_bgr(COLOR_FONDO)

    # Se trazan las líneas que separan el video del encabezado y la barra de controles
    cv2.line(lienzo, (0, ALTO_ENCABEZADO - 1), (ancho, ALTO_ENCABEZADO - 1),
             a_bgr(COLOR_BORDE), 1)
    cv2.line(lienzo, (0, ALTO_ENCABEZADO + alto), (ancho, ALTO_ENCABEZADO + alto),
             a_bgr(COLOR_BORDE), 1)

    # Se resaltan las teclas y se muestran sus funciones en gris
    piezas_pie = []
    for tecla, accion in [("ESPACIO", "pausa"), ("E", "espejo"),
                          ("G", "guarda captura"), ("Q o ESC", "cierra")]:
        if piezas_pie:
            piezas_pie.append(("      ", False, COLOR_SUAVE))
        piezas_pie.append((tecla, True, COLOR_TEXTO))
        piezas_pie.append((" " + accion, False, COLOR_SUAVE))

    escribir(lienzo[0:ALTO_ENCABEZADO], [
        ("Reconocimiento facial en tiempo real", (20, 11), 22, True, COLOR_TITULO),
        (SUBTITULO, (20, 38), 15, False, COLOR_SUAVE),
    ])
    escribir(lienzo[ALTO_ENCABEZADO + alto:],
             textos_en_fila(piezas_pie, 20, 11, 16))

    MARCOS_LISTOS[clave] = lienzo
    return lienzo


def dibujar_marco(vista, rostros, cuadros_por_segundo, pausado):
    """
    Agrega el encabezado y la barra de controles a la ventana,
    con el título del programa, la cantidad de rostros detectados
    y el estado del video.

    En la esquina superior derecha va un punto de color con la palabra que
    dice si el video está corriendo, y debajo los rostros encontrados junto
    con los cuadros por segundo.
    """
    alto, ancho = vista.shape[:2]
    lienzo = marco_base(ancho, alto).copy()
    lienzo[ALTO_ENCABEZADO:ALTO_ENCABEZADO + alto, 0:ancho] = vista

    if pausado:
        estado = "EN PAUSA"
        color_estado = COLOR_DUDA
    else:
        estado = "EN VIVO"
        color_estado = COLOR_ACIERTO

    resumen = "1 rostro" if rostros == 1 else str(rostros) + " rostros"
    if not pausado:
        resumen += " · " + ("%.0f" % cuadros_por_segundo) + " FPS"

    x_estado = ancho - medir_texto(estado, 14, True)[0] - 20
    cv2.circle(lienzo,
               (x_estado - RADIO_PUNTO - 9, 16 + medio_del_texto(estado, 14, True)),
               RADIO_PUNTO, a_bgr(color_estado), -1)

    # Solo se actualizan los dos datos que cambian en el encabezado
    # para evitar procesar toda la ventana en cada fotograma
    escribir(lienzo[0:ALTO_ENCABEZADO], [
        (estado, (x_estado, 16), 14, True, color_estado),
        (resumen, (ancho - medir_texto(resumen, 15)[0] - 20, 37), 15,
         False, COLOR_SUAVE),
    ])
    return lienzo


def guardar_captura(vista):
    """Guarda una captura de la ventana en formato PNG con la fecha y hora"""
    nombre = "captura_" + time.strftime("%Y%m%d_%H%M%S") + ".png"
    ruta = os.path.join(CARPETA_BASE, nombre)
    logrado, datos = cv2.imencode(".png", vista)
    if not logrado:
        print("No se pudo preparar la captura")
        return
    with open(ruta, "wb") as archivo:
        archivo.write(datos.tobytes())
    print("Captura guardada en " + nombre)


def main():
    """Inicializa el modelo y la cámara para realizar el reconocimiento facial en tiempo real."""
    global SUBTITULO

    modelo = cargar_modelo()

    # El modelo indica de qué medida espera las imágenes y cuántas clases tiene
    forma = modelo.input_shape
    lado_entrada = forma[1] if forma[1] else LADO_ENTRADA
    cantidad_clases = modelo.output_shape[-1]
    clases = cargar_clases(cantidad_clases)

    # El subtítulo de la ventana dice cuántas personas distingue la red
    SUBTITULO = "Red neuronal convolucional · " + str(cantidad_clases) + " clases"

    print("Modelo listo con " + str(cantidad_clases) + " clases de " +
          str(lado_entrada) + " × " + str(lado_entrada) + " píxeles")
    print("Clases " + ", ".join(nombre_legible(c) for c in clases))

    cascada = cargar_detector()
    camara = abrir_camara()

    espejo = MODO_ESPEJO
    pausado = False
    seguimientos = []
    cuadros_por_segundo = 0.0
    momento = time.time()

    # Se conserva el último fotograma para mostrarlo durante la pausa
    ultima_imagen = None
    rostros_vistos = 0

    ventana = "Reconocimiento facial en tiempo real con CNN"
    cv2.namedWindow(ventana)

    while True:
        if not pausado:
            logrado, cuadro = camara.read()
            if not logrado:
                print("Se perdió la señal de la cámara")
                break
            if espejo:
                cuadro = cv2.flip(cuadro, 1)

            cajas = buscar_rostros(cuadro, cascada)

            # Se recortan todos los rostros y se clasifican en una sola llamada
            recortes = []
            cajas_cuadradas = []
            for caja in cajas:
                recorte, cuadrada = recortar_rostro(cuadro, caja, lado_entrada)
                recortes.append(recorte)
                cajas_cuadradas.append(cuadrada)

            if recortes:
                lote = preparar_para_la_red(recortes)
                # Se utiliza predict_on_batch para procesar en un solo lote las caras
                # detectadas en cada fotograma, ya que predict realiza operaciones
                # adicionales y, en las pruebas realizadas, tardó aproximadamente
                # nueve veces más, y reducía la fluidez del video
                predicciones = modelo.predict_on_batch(lote)
            else:
                predicciones = []

            # Se asocia cada rostro detectado con su seguimiento correspondiente
            vigentes = []
            for caja, probabilidades in zip(cajas_cuadradas, predicciones):
                x, y, ancho, alto = caja
                centro = (x + ancho // 2, y + alto // 2)
                seguimiento = emparejar(seguimientos, centro, ancho)
                if seguimiento is None:
                    seguimiento = Seguimiento(centro, probabilidades)
                    seguimientos.append(seguimiento)
                else:
                    seguimiento.actualizar(centro, probabilidades)
                vigentes.append((caja, seguimiento))

            # Los seguimientos que llevan varios cuadros sin aparecer se descartan
            for seguimiento in seguimientos:
                if all(seguimiento is not activo for _, activo in vigentes):
                    seguimiento.cuadros_sin_ver += 1
            seguimientos = [s for s in seguimientos if s.cuadros_sin_ver < CUADROS_SUAVIZADO]

            vista = cuadro.copy()
            for caja, seguimiento in vigentes:
                suavizadas = seguimiento.promedio()
                indice = int(np.argmax(suavizadas))
                confianza = float(suavizadas[indice])
                aceptado = confianza >= CONFIANZA_MINIMA
                if aceptado:
                    nombre = nombre_legible(clases[indice])
                else:
                    nombre = "Desconocido"
                dibujar_rostro(vista, caja, nombre, confianza, aceptado)

            ahora = time.time()
            transcurrido = ahora - momento
            momento = ahora
            if transcurrido > 0:
                cuadros_por_segundo = 0.8 * cuadros_por_segundo + 0.2 / transcurrido

            ultima_imagen = vista
            rostros_vistos = len(vigentes)

        if ultima_imagen is not None:
            cv2.imshow(ventana, dibujar_marco(ultima_imagen, rostros_vistos,
                                              cuadros_por_segundo, pausado))

        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (ord("q"), ord("Q"), 27):
            break
        if tecla == ord(" "):
            pausado = not pausado
            momento = time.time()
        if tecla in (ord("e"), ord("E")):
            espejo = not espejo
        if tecla in (ord("g"), ord("G")) and ultima_imagen is not None:
            guardar_captura(dibujar_marco(ultima_imagen, rostros_vistos,
                                          cuadros_por_segundo, pausado))
        if cv2.getWindowProperty(ventana, cv2.WND_PROP_VISIBLE) < 1:
            break

    camara.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
