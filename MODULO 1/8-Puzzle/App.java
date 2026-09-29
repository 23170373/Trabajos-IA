/*
    Alumno: Jonathan Daniel Valencia
    No. de control: 23170373
    Hora: 18:00-19:00
*/

/*
* Clase principal para la ejecución del 8-Puzzle.
* Aquí definimos la configuración de inicio y la meta para probar 
* el rendimiento de los distintos algoritmos de búsqueda a ciegas.
*/
public class App {
    public static void main(String[] args) throws Exception {

        // El tablero se representa como una cadena de 9 caracteres, leída por renglones.
        // El espacio en blanco es la casilla vacía.
        String initialState = "7621 3458"; // Estado inicial del puzzle
        String goalState = "12345678 "; // Estado objetivo del puzzle

        System.out.println("Initial State:");
        System.out.println(NodeUtils.formatState(initialState));
        System.out.println("Goal State:");
        System.out.println(NodeUtils.formatState(goalState));

        SearchTree searchTree = new SearchTree(initialState, goalState);

        searchTree.breadthFirstSearch();
        searchTree.uniformCostSearch();
        searchTree.depthFirstSearch();
        searchTree.depthLimitedSearchStandalone(20);
        searchTree.iterativeDeepeningSearch();
        searchTree.bidirectionalSearch();

        System.out.println("End");
    }
}
