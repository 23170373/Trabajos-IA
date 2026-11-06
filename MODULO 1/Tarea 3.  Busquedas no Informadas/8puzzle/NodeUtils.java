/*
    Alumno: Jonathan Daniel Valencia
    No. de control: 23170373
    Hora: 18:00-19:00
*/

import java.util.ArrayList;
import java.util.List;

/*
* Utilerías del 8-puzzle. Aquí viven la función sucesor y el formato del tablero.
*/
public class NodeUtils {

    // Intercambia dos casillas. Como String es inmutable, se hace sobre un arreglo de caracteres.
    private static String swapPositions(String state, int pos1, int pos2) {
        char[] arr = state.toCharArray();
        char temp = arr[pos1];
        arr[pos1] = arr[pos2];
        arr[pos2] = temp;
        return new String(arr);
    }

    // Función sucesor. Genera todos los estados que se obtienen al mover el espacio vacío a una casilla adyacente.
    public static List<Node> generateChildren(Node parentNode) {

        /*
         * Ejemplo con el espacio vacío en la posición 6:
         *
         *  1 2 3        1 2 3      1 2 3
         *  4 5 6   =>     5 6  +   4 5 6
         *    7 8        4 7 8      7   8
         *
         * "123456 78" => ["123 56478", "1234567 8"]
        */

        List<Node> successors = new ArrayList<>();
        int zeroPos = parentNode.getState().indexOf(" "); // Buscamos dónde está el hueco

        /*
         * Posiciones del tablero:
         *
         * 0 1 2
         * 3 4 5
         * 6 7 8
         *
         * Para cada posición del espacio vacío se indican
         * las posiciones con las que puede intercambiarse.
        */

        int[][] adjacentPositions = {
            {1, 3},           // Índice 0 (Esquina sup izq)
            {0, 2, 4},        // Índice 1 (Borde sup)
            {1, 5},           // Índice 2 (Esquina sup der)
            {0, 4, 6},        // Índice 3 (Borde izq)
            {1, 3, 5, 7},     // Índice 4 (Centro - todos los movimientos)
            {2, 4, 8},        // Índice 5 (Borde der)
            {3, 7},           // Índice 6 (Esquina inf izq)
            {4, 6, 8},        // Índice 7 (Borde inf)
            {5, 7}            // Índice 8 (Esquina inf der)
        };

        // Cada hijo se crea con referencia a su nodo padre. Esa trazabilidad
        // es la que permite reconstruir la ruta de solución.
        for (int adjPos : adjacentPositions[zeroPos]) {
            String newState = swapPositions(parentNode.getState(), zeroPos, adjPos);
            successors.add(new Node(newState, parentNode));
        }

        return successors;
    }

    // Convierte la cadena de 9 caracteres en un tablero de 3 x 3 para imprimirlo.
    public static String formatState(String state){
        String formattedState = "";

        for(int i = 0; i < state.length(); i++){
            formattedState += state.charAt(i) + " ";
            if((i + 1) % 3 == 0){
                formattedState += "\n";
            }
        }
        
        return formattedState;
    }
}
