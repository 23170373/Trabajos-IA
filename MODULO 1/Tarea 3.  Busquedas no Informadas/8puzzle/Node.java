/*
    Alumno: Jonathan Daniel Valencia
    No. de control: 23170373
    Hora: 18:00-19:00
*/

/*
 * Nodo del árbol de búsqueda.
 * Representa un estado del 8-puzzle junto con la información necesaria
 * para recorrer el árbol y reconstruir la ruta de solución.
 */
public class Node {
    // Tablero como cadena de 9 caracteres. El espacio en blanco es la casilla vacía.
    private String state;
    // Nodo padre. Permite regresar desde el estado objetivo hasta el estado inicial.
    private Node parent;
    // Nivel del nodo dentro del árbol de búsqueda.
    private int depth;
    // Costo acumulado desde el nodo raíz. Lo utiliza la búsqueda de costo uniforme.
    private int cost;

    public Node(String state, Node parent) {
        this.state = state;
        this.parent = parent;
        
        // El nodo raíz tiene profundidad 0 y cada hijo queda un nivel por debajo de su padre.
        // Sin esta asignación todos los nodos se quedaban en profundidad 0.
        // La raíz inicia en 0 y cada hijo suma 1 al nivel anterior.
        if (parent != null) {
            this.depth = parent.getDepth() + 1; 
        }
    }

    public String getState() {
        return state;
    }

    public void setState(String state) {
        this.state = state;
    }

    public Node getParent() {
        return parent;
    }

    public void setParent(Node parent) {
        this.parent = parent;
    }

    public int getDepth() {
        return depth;
    }

    public void setDepth(int depth) {
        this.depth = depth;
    }

    public int getCost() {
        return cost;
    }

    public void setCost(int cost) {
        this.cost = cost;
    }
}
