/*
    Alumno: Jonathan Daniel Valencia
    No. de control: 23170373
    Hora: 18:00-19:00
*/

/*
 * Clase que define la estructura basica de un elemento dentro del arbol.
 * Cada nodo almacena un valor (en este caso un nombre) y contiene los punteros
 * que lo conectan con sus posibles nodos hijos (izquierdo y derecho).
*/

public class Nodo {
    String nombre;
    Nodo izquierdo;
    Nodo derecho;

    public Nodo(String nombre) {
        this.nombre = nombre;
        this.izquierdo = null;
        this.derecho = null;
    }
}