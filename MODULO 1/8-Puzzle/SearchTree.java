/*
    Alumno: Jonathan Daniel Valencia
    No. de control: 23170373
    Hora: 18:00-19:00
*/

import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;

/*
 * Árbol de búsqueda del 8-puzzle.
 * Contiene las búsquedas no informadas vistas en clase (primero en anchura,
 * primero en profundidad y costo uniforme). Todas parten del mismo nodo raíz.
*/

public class SearchTree {
    Node root;
    String goalState; // Variable de texto donde guardaremos el estado o meta que buscamos, para no tener que pasarla como parámetro a cada función de búsqueda.
    String initialState; // Variable donde guardaremos como se ve el problema al inicio.
    public String resumenGlobal = "\n========== RESUMEN FINAL DE ESTADÍSTICAS ==========\n"; // Variable para guardar los resultados y mostrarlos al final

    public SearchTree(String initialState, String goalState) {
        this.initialState = initialState;
        this.goalState = goalState;

        // El nodo raíz (punto de partida) no tiene padre, así que se pasa null. Su profundidad es 0 y su costo acumulado también.
        this.root = new Node(initialState, null);
    }

    // ==========================================================
    // ----------- Búsqueda primero en anchura (BFS). -----------
    // ==========================================================

    // Explora el árbol por niveles usando una cola FIFO.

    public void breadthFirstSearch() {
        System.out.println("===== Breadth First Search =====");
        // Cuenta los nodos procesados durante esta ejecución.
        int time = 0;

        // Estados visitados, para no volver a procesar un tablero ya explorado.
        Set<String> visited = new HashSet<String>();
        Node currentNode = root;

        // BFS utiliza una cola FIFO para explorar nivel por nivel uniformemente
        Queue<Node> queue = new LinkedList<>();
        queue.add(currentNode);
        
        // Continúa la búsqueda mientras existan nodos pendientes por explorar.
        while (!queue.isEmpty()) {
            time++;
            // Extrae el nodo más antiguo de la cola (FIFO) y lo marca como visitado.
            currentNode = queue.poll();
            visited.add(currentNode.getState());

            //System.out.println(NodeUtils.formatState(currentNode.getState()));

            // Verificamos si llegamos a la solución
            if(currentNode.getState().equals(goalState)) {
                System.out.println("Goal state found: " + currentNode.getState());
                printPath(currentNode);
                break;
            }

            // Expandimos los nodos sucesores que no hayan sido visitados aún
            List<Node> children = NodeUtils.generateChildren(currentNode);
            for (Node child : children) {
                if (!visited.contains(child.getState()))
                    queue.add(child);
            }
        }

        // Nodos procesados, estados visitados y tamaño final de la cola.
        System.out.println("--- Datos estadisticos ---");
        System.out.println("Time: " + time);
        System.out.println("Estados visitados: " + visited.size());
        System.out.println("Queue: " + queue.size());
        System.out.println();

        resumenGlobal += "1. Anchura (BFS)       -> Time: " + time + " | Visitados: " + visited.size() + " | Cola (Queue): " + queue.size() + "\n"; // Guarda los resultados
    }

    // ==========================================================
    // --------- Búsqueda primero en profundidad (DFS). ---------
    // ==========================================================

    // Sigue una rama hasta el fondo antes de probar otra.
    // Mismo procedimiento que BFS, pero con una pila (LIFO) en lugar de una cola.
    public void depthFirstSearch() {
        System.out.println("--- Depth First Search ---");
        int time = 0;

        Set<String> visited = new HashSet<String>();
        Node currentNode = root;

        // DFS utiliza una Pila LIFO, lo que obliga a explorar hasta el fondo de una rama
        Stack<Node> stack = new Stack<>();
        stack.push(currentNode);

        while (!stack.isEmpty()) {
            time++;

            // Extrae el último nodo agregado a la pila (LIFO) y lo marca como visitado.
            currentNode = stack.pop();
            visited.add(currentNode.getState());

            // Test objetivo.
            if(currentNode.getState().equals(goalState)) {
                System.out.println("Goal state found: " + currentNode.getState());
                printPath(currentNode);
                break;
            }

            // Apila los sucesores no visitados. El último apilado será el siguiente en procesarse.
            List<Node> children = NodeUtils.generateChildren(currentNode);
            for (Node child : children) {
                if (!visited.contains(child.getState()))
                    stack.push(child);
            }
        }

        System.out.println("--- Datos estadisticos ---");
        System.out.println("Time: " + time);
        System.out.println("Estados visitados: " + visited.size());
        System.out.println("Stack: " + stack.size());
        System.out.println();

        resumenGlobal += "2. Profundidad (DFS)   -> Time: " + time + " | Visitados: " + visited.size() + " | Pila (Stack): " + stack.size() + "\n"; // Guarda los resultados
    }

    // ==========================================================
    // ----------- Búsqueda de costo uniforme (UCS). -----------
    // ==========================================================

    // Expande primero el nodo con menor costo acumulado usando una cola de prioridad.
    // Como cada movimiento cuesta 1, se comporta igual que BFS.
    public void uniformCostSearch() {
        System.out.println("*** Uniform Cost Search ***");
        int time = 0;

        Set<String> visited = new HashSet<String>();
        Node currentNode = root; // El nodo raíz entra con costo 0.

        PriorityQueue<Node> queue = new PriorityQueue<>(new NodePriorityComparator());
        queue.add(currentNode);

        while (!queue.isEmpty()) {
            time++;

            // Extrae el nodo de menor costo acumulado y lo marca como visitado.
            currentNode = queue.poll();
            visited.add(currentNode.getState());

            // Test objetivo.
            if(currentNode.getState().equals(goalState)) {
                System.out.println("Goal state found: " + currentNode.getState());
                printPath(currentNode);
                System.out.println("Costo de la solucion: " + currentNode.getCost());
                break;
            }

            List<Node> children = NodeUtils.generateChildren(currentNode);
            for (Node child : children) {
                if (!visited.contains(child.getState())) {
                    // Cada movimiento cuesta 1, así que el hijo hereda el costo del padre más 1.
                    child.setCost(child.getParent().getCost() + 1);
                    queue.add(child);
                }
            }
        }
        
        System.out.println("--- Datos estadisticos ---");
        System.out.println("Time: " + time);
        System.out.println("Estados visitados: " + visited.size());
        System.out.println("Priority Queue: " + queue.size());
        System.out.println();

        resumenGlobal += "3. Costo Uniforme (UCS)-> Time: " + time + " | Visitados: " + visited.size() + " | Prioridad (PQ): " + queue.size() + "\n"; // Guarda los resultados
    }

    // Recorre primero los nodos padre para imprimir la solución
    // desde el estado inicial hasta el estado objetivo, y no al revés.
    public void printPath(Node node) {
        if (node == null) {
            return;
        }
        printPath(node.getParent());
        System.out.println(NodeUtils.formatState(node.getState()));
    }
}
