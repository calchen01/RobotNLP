from collections import defaultdict

# This class will be used by r2d2_commands.py for navigation
class Graph:
    def __init__(self, grid):
        self.V = set() # Set of vertices, a vertex is a coordinate in our grid
        self.E = set() # Set of edges, an edge connects two vertices

        for row in range(len(grid)):
            for col in range(len(grid[0])):
                self.V.add((row, col)) # Add the coordinate to the set of vertices
                # Add the edge connecting the start and target vertices (both directions) to the set of edges
                # if the target vertex is in the grid and both the start and target vertices are not obstacles
                if row - 1 >= 0 and grid[row][col] in {"", "you", "target"} and grid[row - 1][col] in {"", "you", "target"}:
                    self.E.add(((row - 1, col), (row, col)))
                    self.E.add(((row, col), (row - 1, col)))
                if row + 1 < len(grid) and grid[row][col] in {"", "you", "target"} and grid[row + 1][col] in {"", "you", "target"}:
                    self.E.add(((row + 1, col), (row, col)))
                    self.E.add(((row, col), (row + 1, col)))
                if col - 1 >= 0 and grid[row][col] in {"", "you", "target"} and grid[row][col - 1] in {"", "you", "target"}:
                    self.E.add(((row, col - 1), (row, col)))
                    self.E.add(((row, col), (row, col - 1)))
                if col + 1 < len(grid[0]) and grid[row][col] in {"", "you", "target"} and grid[row][col + 1] in {"", "you", "target"}:
                    self.E.add(((row, col + 1), (row, col)))
                    self.E.add(((row, col), (row, col + 1)))
                
        neighborhood = defaultdict(set)
        for (u, v) in self.E:
            neighborhood[u].add(v)
            neighborhood[v].add(u) # assumes undirected graph
        self.neighborhood = neighborhood

    def neighbors(self, u):
        return self.neighborhood[u]

    def dist_between(self, u, v):
        if v not in self.neighbors(u):
            return None
        return 1 # assumes unweighted graph

    def print(self):
        print(self.V)
        print(self.E)