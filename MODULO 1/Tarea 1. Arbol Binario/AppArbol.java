/*
    Alumno: Jonathan Daniel Valencia
    No. de control: 23170373
    Hora: 18:00-19:00
*/

/*
 * Clase principal de ejecucion (App).
 * Su funcion es instanciar el arbol binario, construir una estructura de prueba
 * agregando nodos manualmente, y verificar el correcto funcionamiento de los
 * metodos vacio() y buscarNodo().
*/

public class AppArbol {
    public static void main(String[] args) {

        Arbol miArbol = new Arbol(); // Instanciamos el arbol
        String objetivo = "D";
        Nodo resultado;

        // Probamos el metodo vacio() antes de agregar datos
        System.out.println("¿El arbol esta vacio? " + miArbol.vacio());

        // Creamos nodos manualmente para armar un arbol pequeño de prueba
        miArbol.raiz = new Nodo("A");
        miArbol.raiz.izquierdo = new Nodo("B");
        miArbol.raiz.derecho = new Nodo("C");
        miArbol.raiz.izquierdo.izquierdo = new Nodo("D");

        /*
            Representacion visual:
                                    A
                                   /\
                                  /  \
                                 B    C
                                /
                               /
                              "D" <-- Objetivo 
        */

        // Probamos el metodo vacio() despues de agregar datos
        System.out.println("¿El arbol esta vacio ahora? " + miArbol.vacio());

        // Probamos el metodo buscar nodo
        System.out.println("\nBuscando el nodo: " + objetivo);
        resultado = miArbol.buscarNodo(objetivo);

        // Verificamos si se encontro
        if (resultado != null) {
            System.out.println("Nodo encontrado con el nombre: " + resultado.nombre);
        } else {
            System.out.println("El nodo no existe en el arbol");
        }
    }
}