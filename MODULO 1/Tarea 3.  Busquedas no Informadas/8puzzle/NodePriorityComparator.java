import java.util.Comparator;
/*
    Alumno: Jonathan Daniel Valencia
    No. de control: 23170373
    Hora: 18:00-19:00
*/

/*
 * Comparator utilizado por la PriorityQueue de la búsqueda de costo uniforme.
 * Ordena los nodos por costo acumulado, de menor a mayor, de modo que
 * la cola de prioridad siempre entregue primero el nodo más barato.
 */
public class NodePriorityComparator implements Comparator<Node> {

    @Override
    public int compare(Node n1, Node n2) {
        return Integer.compare(n1.getCost(), n2.getCost());
    }
}
