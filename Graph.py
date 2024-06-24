from collections import deque

class Node:
    def __init__(self, id):
        self.id = id
        self.edges = []

class Edge:
    def __init__(self, src, dest, capacity):
        """
        创建一个边对象。

        参数：
        src (int): 边的起始节点。
        dest (int): 边的目标节点。
        capacity (int): 边的容量。

        属性：
        src (int): 边的起始节点。
        dest (int): 边的目标节点。
        capacity (int): 边的容量。
        flow (int): 边的流量，默认为0。
        reverse (Edge): 边的反向边，默认为None。
        """
        self.src = src
        self.dest = dest
        self.capacity = capacity
        self.flow = 0
        self.reverse = None

class Graph:
    def __init__(self, size):
        """
        创建一个图对象。

        参数：
        size (int): 图的大小。

        属性：
        nodes (list): 图中的节点列表。
        size (int): 图的大小。
        """
        self.nodes = [Node(i) for i in range(size)]
        self.size = size

    def add_edge(self, src, dest, capacity):
        """
        添加一条边到图中。

        参数：
        src (int): 边的起始节点。
        dest (int): 边的目标节点。
        capacity (int): 边的容量。
        """
        forward_edge = Edge(src, dest, capacity)
        reverse_edge = Edge(dest, src, capacity)
        forward_edge.reverse = reverse_edge
        reverse_edge.reverse = forward_edge
        self.nodes[src].edges.append(forward_edge)
        self.nodes[dest].edges.append(reverse_edge)

    def bfs(self, source, sink, parent):
        """
        使用广度优先搜索算法查找从源节点到汇点的路径。

        参数：
        source (int): 源节点。
        sink (int): 汇点。
        parent (list): 记录节点的父节点。

        返回：
        bool: 如果存在从源节点到汇点的路径，则返回True；否则返回False。
        """
        visited = set()
        queue = deque([source])
        visited.add(source)

        while queue:
            current = queue.popleft()
            for edge in self.nodes[current].edges:
                if edge.dest not in visited and edge.capacity - edge.flow > 0:
                    parent[edge.dest] = edge
                    visited.add(edge.dest)
                    if edge.dest == sink:
                        return True
                    queue.append(edge.dest)
        return False

    def min_cut_set(self, source):
        """
        查找最小割集合。

        参数：
        source (int): 源节点。

        返回：
        list: 最小割集合，包含割边的起始节点、目标节点和剩余容量。
        """
        visited = set()
        queue = deque([source])
        visited.add(source)

        while queue:
            current = queue.popleft()
            for edge in self.nodes[current].edges:
                if edge.dest not in visited and edge.capacity - edge.flow > 0:
                    visited.add(edge.dest)
                    queue.append(edge.dest)

        min_cut_edges = []
        for u in visited:
            for edge in self.nodes[u].edges:
                if edge.dest not in visited and edge.capacity > 0:
                    min_cut_edges.append((u, edge.dest, edge.capacity - edge.flow))

        return min_cut_edges

    def find_max_flow(self, source, sink):
        """
        查找最大流量。

        参数：
        source (int): 源节点。
        sink (int): 汇点。

        返回：
        int: 最大流量。
        """
        parent = [None] * self.size
        max_flow = 0

        while self.bfs(source, sink, parent):
            path_flow = float('Inf')
            s = sink
            while s != source:
                path_flow = min(path_flow, parent[s].capacity - parent[s].flow)
                s = parent[s].src

            v = sink
            while v != source:
                parent[v].flow += path_flow
                parent[v].reverse.flow -= path_flow
                v = parent[v].src

            max_flow += path_flow

        return max_flow

if __name__ == "__main__":
    g = Graph(5)
    g.add_edge(0, 1, 10)
    g.add_edge(1, 2, 9)
    g.add_edge(2, 3, 10)
    g.add_edge(0, 2, 10)
    g.add_edge(1, 3, 7)
    g.add_edge(3, 4, 10)
    g.add_edge(2, 4, 8)

    max_flow = g.find_max_flow(0, 4)
    min_cut_edges = g.min_cut_set(0)
    print(f"最大流量: {max_flow}")
    print("最小割集合:", min_cut_edges)

