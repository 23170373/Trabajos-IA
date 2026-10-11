/*
    Alumno: Jonathan Daniel Valencia
    No. de control: 23170373
    Hora: 18:00-19:00
*/

/*
 * Clase que representa la estructura de datos del Arbol Binario.
 * Se encargada de administrar el nodo raiz y proporciona los metodos
 * para interactuar con el arbol, como comprobar si esta vacio o buscar
 * un elemento especifico mediante un recorrido en profundidad (Pre-orden).
*/

public class Arbol {
    Nodo raiz; // Variable que guarda el punto de inicio del arbol

    // Constructor
    public Arbol() {
        this.raiz = null;
    }

    // Metodo para revisar si la raiz sigue estando vacia, si es "true" esta vacio
    public boolean vacio() {
        return this.raiz == null;
    }

    // Metodo para indicar que nombre u objetivo buscar empezando desde la raiz
    public Nodo buscarNodo(String nombre) {
        return buscarAux(this.raiz, nombre);
    }

    // Metodo de busqueda
    private Nodo buscarAux(Nodo actual, String nombre) {

        // Caso base (parada de emergencia si llegamos a un camino sin salida o nodo inexistente)
        if (actual == null) {
            return null;
        }

        // Revisamos el nodo actual
        if (actual.nombre.equals(nombre)) {
            return actual; // Si los nombres coinciden se devuelve este nodo y la busqueda termina
        }

        // Buscamos por la izquierda
        Nodo encontradoIzquierda = buscarAux(actual.izquierdo, nombre);

        // Si la busqueda por la izquierda encuentra algo devuelve el resultado hacia arriba (Ejemplo: D --> B --> A --> Clase: App)
        if (encontradoIzquierda != null) {
            return encontradoIzquierda;
        }

        // Buscamos por la derecha
        return buscarAux(actual.derecho, nombre);
    }
}
