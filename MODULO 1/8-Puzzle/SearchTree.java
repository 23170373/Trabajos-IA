/*
    Alumno: Jonathan Daniel Valencia
    No. de control: 23170373
    Hora: 18:00-19:00
*/

import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
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
        System.out.println("\n===== Breadth First Search =====");
        // Contador para nodos procesados
        int time = 0;

        // Iniciamos el cronometro y la variable de pico maximo
        long startTime = System.nanoTime();
        int maxQueueSize = 0;

        // Estados visitados, para no volver a procesar un tablero ya explorado.
        Set<String> visited = new HashSet<String>();
        Node currentNode = root;

        // BFS utiliza una cola FIFO para explorar nivel por nivel uniformemente
        Queue<Node> queue = new LinkedList<>();
        queue.add(currentNode);
        
        // Se continua la busqueda mientras existan nodos pendientes por explorar
        while (!queue.isEmpty()) {
            time++;

            // En cada vuelta revisamos si la cola supero su record historico de tamaño
            if (queue.size() > maxQueueSize) {
                maxQueueSize = queue.size();
            }
            
            // Extraemos el nodo mas antiguo de la cola (FIFO) y se marca como visitado
            currentNode = queue.poll();
            visited.add(currentNode.getState());

            //System.out.println(NodeUtils.formatState(currentNode.getState()));

            // Verificamos si llegamos a la solucion
            if(currentNode.getState().equals(goalState)) {
                // Detenemos el reloj y calculamos los milisegundo
                long endTime = System.nanoTime();
                double timeInMs = (endTime - startTime) / 1_000_000.0;

                System.out.println("Goal state found: " + currentNode.getState());
                //printPath(currentNode); // Impresion del proceso de ejecucion
                System.out.println("Profundidad de la solucion: " + currentNode.getDepth());
                
                // Imprimimos las estadisticas o resultados
                System.out.println(); // Espacio para facil lectura
                System.out.println("--- Datos estadisticos ---");
                System.out.println("Time: " + time);
                System.out.println("Estados visitados: " + visited.size());
                System.out.println("Queue: " + queue.size());
                System.out.println("Max Queue: " + maxQueueSize);
                System.out.println("Tiempo real (ms): " + String.format("%.4f", timeInMs));
                break;
            }

            // Expandimos los nodos sucesores que no hayan sido visitados aun
            List<Node> children = NodeUtils.generateChildren(currentNode);
            for (Node child : children) {
                if (!visited.contains(child.getState()))
                    queue.add(child);
            }
        }

    }

    // ==========================================================
    // --------- Búsqueda primero en profundidad (DFS). ---------
    // ==========================================================

    // Se sigue una rama hasta el final antes de pasar a otra
    // Mismo procedimiento que BFS, pero con una pila (LIFO) en lugar de una cola
    public void depthFirstSearch() {
        System.out.println("\n==== Depth First Search ====");
        int time = 0;

        long startTime = System.nanoTime();
        int maxStackSize = 0;

        Set<String> visited = new HashSet<String>();
        Node currentNode = root;

        // DFS utiliza una Pila LIFO, lo que obliga a explorar hasta el fondo de una rama
        Stack<Node> stack = new Stack<>();
        stack.push(currentNode);

        while (!stack.isEmpty()) {
            if (stack.size() > maxStackSize) {
                maxStackSize = stack.size();
            }

            time++;

            // Se extrae el ultimo nodo agregado a la pila (LIFO) y se marca como visitado
            currentNode = stack.pop();
            visited.add(currentNode.getState());

            // Verificamos si este nodo es la solucion final
            if(currentNode.getState().equals(goalState)) {
                System.out.println("Goal state found: " + currentNode.getState());
                System.out.println("Profundidad de la solucion: " + currentNode.getDepth());
                //printPath(currentNode); // Impresion del proceso de ejecucion

                long endTime = System.nanoTime();
                double timeInMs = (endTime - startTime) / 1_000_000.0;
                System.out.println(); // Espacio para facil lectura
                System.out.println("--- Datos estadisticos ---");
                System.out.println("Time: " + time);
                System.out.println("Estados visitados: " + visited.size());
                System.out.println("Stack: " + stack.size());
                System.out.println("Max Stack: " + maxStackSize);
                System.out.println("Tiempo real (ms): " + String.format("%.4f", timeInMs));

                break;
            }

            // Apilamos los sucesores no visitados y el ultimo en apilarse sera el siguiente en ser procesado
            List<Node> children = NodeUtils.generateChildren(currentNode);
            for (Node child : children) {
                if (!visited.contains(child.getState()))
                    stack.push(child);
            }
        }

    }

    // ==========================================================
    // ----------- Búsqueda de costo uniforme (UCS). -----------
    // ==========================================================

    // Se expande primero el nodo con menor costo acumulado usando una cola de prioridad
    // Como cada movimiento cuesta 1, se comporta igual que BFS
    public void uniformCostSearch() {
        System.out.println("\n==== Uniform Cost Search ====");
        int time = 0;

        long startTime = System.nanoTime();
        int maxQueueSize = 0;

        Set<String> visited = new HashSet<String>();
        Node currentNode = root; // El nodo raiz entra con costo 0

        PriorityQueue<Node> queue = new PriorityQueue<>(new NodePriorityComparator());
        queue.add(currentNode);

        while (!queue.isEmpty()) {
            time++;

            if (queue.size() > maxQueueSize) {
                maxQueueSize = queue.size();
            }

            // Se extrae el nodo de menor costo acumulado y se marca como visitado
            currentNode = queue.poll();
            visited.add(currentNode.getState());

            // Verificamos si este nodo es la solucion final
            if(currentNode.getState().equals(goalState)) {
                System.out.println("Goal state found: " + currentNode.getState());
                //printPath(currentNode); // Imprimimos el procedimiento
                System.out.println("Costo de la solucion: " + currentNode.getCost());

                long endTime = System.nanoTime();
                double timeInMs = (endTime - startTime) / 1_000_000.0;
                System.out.println(); // Espacio para facil lectura
                System.out.println("--- Datos estadisticos ---");
                System.out.println("Time: " + time);
                System.out.println("Estados visitados: " + visited.size());
                System.out.println("Priority Queue: " + queue.size());
                System.out.println("Max Priority Queue: " + maxQueueSize);
                System.out.println("Tiempo real (ms): " + String.format("%.4f", timeInMs));

                break;
            }

            List<Node> children = NodeUtils.generateChildren(currentNode);
            for (Node child : children) {
                if (!visited.contains(child.getState())) {
                    // Cada movimiento cuesta 1, asi que le sumamos al hijo el costo del padre mas 1
                    child.setCost(child.getParent().getCost() + 1);
                    queue.add(child);
                }
            }
        }

    }

    // ==========================================================
    // ---- Búsqueda en Profundidad Limitada (Independiente) ----
    // ==========================================================

    // Probamos Búsqueda en Profundidad Limitada por separado para obtener los resultados necesarios para realizar la tabla comparativa
    // y compararla con los demas algoritmos de forma individual

    // Metodo publico para ejecutar y medir DLS de forma aislada de IDS
    public void depthLimitedSearchStandalone(int limit) {
        System.out.println("\n===== Depth Limited Search (Límite " + limit + ") =====");
        
        long startTime = System.nanoTime();
        Set<String> visited = new HashSet<>();
        int[] time = {0}; // Agregamos el contador

        Node result = depthLimitedSearch(root, limit, visited, time); // Usamos el motor con contador

        long endTime = System.nanoTime();
        double timeInMs = (endTime - startTime) / 1_000_000.0;

        if (result != null) {
            System.out.println("Goal state found: " + result.getState());
            //printPath(result); // Impresion del proceso de ejecucion
            System.out.println("Profundidad de la solucion: " + result.getDepth());

            System.out.println(); // Espacio para facil lectura
            System.out.println("--- Datos estadisticos ---");
            System.out.println("Time: " + time[0]); 
            System.out.println("Profundidad maxima alcanzada: " + result.getDepth());
            System.out.println("Tiempo real (ms): " + String.format("%.4f", timeInMs));
            
        } else {
            System.out.println("No se encontró solución dentro del límite de " + limit);
        }
    }

    // ===========================================================
    // ------- Búsqueda en Profundidad Limitada e Iterativa ------
    // ===========================================================

    // Combinamos ambos algoritmos (DLS + IDS)
    // Version recursiva del algoritmo DLS (Depth-Limited Search).
    // Se utiliza recursividad en lugar de una estructura Stack para poder aplicar backtracking
    // de forma eficiente y evitar el desbordamiento de memoria ante los ciclos del 8-Puzzle
    private Node depthLimitedSearch(Node currentNode, int limit, Set<String> visited, int[] counter) {
        counter[0]++; // Incrementamos el contador en 1 por cada nodo que procesamos
        
        if (currentNode.getState().equals(goalState)) {
            return currentNode;
        }
        
        if (currentNode.getDepth() >= limit) {
            return null;
        }

        visited.add(currentNode.getState());
        
        List<Node> children = NodeUtils.generateChildren(currentNode);
        for (Node child : children) {
            if (!visited.contains(child.getState())) {
                Node result = depthLimitedSearch(child, limit, visited, counter);
                if (result != null) {
                    return result; 
                }
            }
        }
        
        visited.remove(currentNode.getState());
        return null;
    }

    // ------- Algoritmo IDS (Iterative Deepening Search). -------
    
    public void iterativeDeepeningSearch() {
        System.out.println("\n===== Iterative Deepening Search =====");
        int maxLimit = 30; 
        Node result = null;
        int totalTimeAllIterations = 0; // Acumulador del total de nodos

        long startTime = System.nanoTime();

        for (int limit = 0; limit <= maxLimit; limit++) {
            Set<String> visited = new HashSet<>();
            int[] iterationTime = {0}; // Arreglo para contar solo los nodos de este limite
            
            result = depthLimitedSearch(root, limit, visited, iterationTime);
            totalTimeAllIterations += iterationTime[0]; // Sumamos al acumulador global
            
            // Imprimimos la cantidad de nodos visitados
            System.out.println("Limite " + limit + " - Time: " + iterationTime[0]);
            
            if (result != null) {
                System.out.println("Goal state found: " + result.getState());
                //printPath(result); // Impresion del proceso de ejecucion
                System.out.println("Profundidad de la solucion: " + result.getDepth());
                break;
            }
        }
        
        long endTime = System.nanoTime();
        double timeInMs = (endTime - startTime) / 1_000_000.0;

        if (result != null) {
            System.out.println(); // Espacio para facil lectura
            System.out.println("--- Datos estadisticos ---");
            System.out.println("Time (todas las iteraciones): " + totalTimeAllIterations);
            System.out.println("Profundidad maxima alcanzada: " + result.getDepth());
            System.out.println("Tiempo real (ms): " + String.format("%.4f", timeInMs));
        } else {
            System.out.println("No se encontró solución dentro del límite.");
        }
        System.out.println();

    }

    // ==========================================================
    // -------------- Búsqueda Bidireccional (BDS) --------------
    // ==========================================================

    public void bidirectionalSearch() {
        System.out.println("\n===== Bidirectional Search =====");
        
        Queue<Node> forwardQueue = new LinkedList<>();
        Queue<Node> backwardQueue = new LinkedList<>();

        Map<String, Node> forwardVisited = new HashMap<>();
        Map<String, Node> backwardVisited = new HashMap<>();

        // Nodo meta inicializado sin padre para expandirse hacia atras
        Node goalRoot = new Node(goalState, null);
        
        forwardQueue.add(root);
        forwardVisited.put(root.getState(), root);
        
        backwardQueue.add(goalRoot);
        backwardVisited.put(goalRoot.getState(), goalRoot);

        // Arreglo para guardar los dos nodos que chocan en el centro: [0] Forward, [1] Backward
        Node[] meeting = null;
        int []time = {0};

        long startTime = System.nanoTime();
        int maxQueueSize = 0;

        // Validacion inicial por si el estado raiz ya es la meta
        if (root.getState().equals(goalState)) {
            meeting = new Node[] { root, goalRoot };
        }

        while (meeting == null && !forwardQueue.isEmpty() && !backwardQueue.isEmpty()) {
            
            // Le pasamos la variable 'time'
            meeting = expandLevel(forwardQueue, forwardVisited, backwardVisited, time);
            
            // Medimos el pico de la cola inmediatamente despues de expandir
            int currentFrontier = forwardQueue.size() + backwardQueue.size();
            if (currentFrontier > maxQueueSize) maxQueueSize = currentFrontier;

            if (meeting == null) {
                // Le pasamos la variable 'time'
                meeting = expandLevel(backwardQueue, backwardVisited, forwardVisited, time);
                
                currentFrontier = forwardQueue.size() + backwardQueue.size();
                if (currentFrontier > maxQueueSize) maxQueueSize = currentFrontier;
                
                if (meeting != null) {
                    meeting = new Node[] { meeting[1], meeting[0] };
                }
            }
        }

        if (meeting != null) {
            Node fromStart = meeting[0];
            Node fromGoal = meeting[1];
            int totalDepth = fromStart.getDepth() + fromGoal.getDepth();

            System.out.println("Goal state found: " + goalState);
            System.out.println("Estado de encuentro: " + fromStart.getState());
            System.out.println("Profundidad de la solucion: " + totalDepth);
            
            //printPath(fromStart);
            
            /* 
            // Imprimimos desde el choque hasta la meta (subiendo por los padres del lado objetivo)
            Node node = fromGoal.getParent();
            while (node != null) {
                // Le damos forma de cuadricula
                System.out.println(NodeUtils.formatState(node.getState()) + "\n");
                node = node.getParent();
            }
            */

            long endTime = System.nanoTime();
            double timeInMs = (endTime - startTime) / 1_000_000.0;
            System.out.println(); // Espacio para facil lectura
            System.out.println("--- Datos estadisticos ---");
            System.out.println("Time: " + time[0]);
            System.out.println("Estados visitados (Inicio+Meta): " + (forwardVisited.size() + backwardVisited.size()));
            System.out.println("Queue (ambas): " + (forwardQueue.size() + backwardQueue.size()));
            System.out.println("Max Queue (ambas): " + maxQueueSize);
            System.out.println("Tiempo real (ms): " + String.format("%.4f", timeInMs));

        } else {
            System.out.println("No se encontró intersección.");
        }
        System.out.println();
    }

    // Metodo auxiliar para expandir un nivel completo a la vez y buscar intersecciones
    // Añadimos el parámetro int[] timeCounter
    private Node[] expandLevel(Queue<Node> queue, Map<String, Node> visitedThisSide, Map<String, Node> visitedOtherSide, int[] timeCounter) {
        int levelSize = queue.size();

        for (int i = 0; i < levelSize; i++) {
            Node currentNode = queue.poll();
            timeCounter[0]++; // Sumamos cada nodo real procesado
            
            List<Node> children = NodeUtils.generateChildren(currentNode);
            
            for (Node child : children) {
                Node other = visitedOtherSide.get(child.getState());
                if (other != null) {
                    // Si los caminos chocan, detenemos la busqueda y entregamos ambas mitades
                    return new Node[] { child, other };
                }

                if (!visitedThisSide.containsKey(child.getState())) {
                    visitedThisSide.put(child.getState(), child);
                    queue.add(child);
                }
            }
        }
        return null;
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
