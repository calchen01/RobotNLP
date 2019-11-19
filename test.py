import re
from graph import *
from a_star import *

class Robot:
    def __init__(self):
        self.name = "R2"
        self.grid = [[]]
        self.pos = (-1, -1)

    def extractCoord(self, arr):
        ret = []
        for x in arr:
            if x.isdigit():
                ret.append(int(x))
                break
        for i in range(len(arr) - 1, -1, -1):
            if arr[i].isdigit():
                ret.append(int(arr[i]))
                break
        return ret

    def extractObj(self, arr):
        ret = ""
        ind1 = -1
        ind2 = -1
        for i in range(len(arr)):
            if arr[i] in {"is", "are"}:
                ind1 = i
            elif arr[i] == "at":
                ind2 = i
        for i in range(ind1 + 1, ind2):
            ret += arr[i]
            ret += " "
        return ret[:-1]

    def gridParser(self, command):
        if re.search("\d+ ?(x|by) ?\d+", command):
            arr = re.split("(x|[^a-zA-Z0-9])", command) # guaranteed to contain 2 numbers
            x, y = self.extractCoord(arr)
            self.grid = [["" for col in range(y)] for row in range(x)]
            for row in self.grid:
                print(row)
            return True
        elif re.search("(is|are) .+ at [(]?\d+ ?, ?\d+", command):
            arr = re.split("[^a-zA-Z0-9]", command) # guaranteed to contain 2 numbers
            x, y = self.extractCoord(arr)
            obj = self.extractObj(arr)
            if len(self.grid) == 0 or len(self.grid[0]) == 0:
                print("grid is not initialized yet")
                return False
            if x < 0 or x >= len(self.grid) or y < 0 or y >= len(self.grid[0]):
                print("coordinate is out of grid")
                return False
            self.grid[x][y] = obj
            for row in self.grid:
                print(row)
            return True
        elif re.search("you are at [(]?\d+ ?, ?\d+", command):
            arr = re.split("[^a-zA-Z0-9]", command) # guaranteed to contain 2 numbers
            x, y = self.extractCoord(arr)
            if len(self.grid) == 0 or len(self.grid[0]) == 0:
                print("grid is not initialized yet")
                return False
            if x < 0 or x >= len(self.grid) or y < 0 or y >= len(self.grid[0]):
                print("coordinate is out of grid")
                return False
            self.pos = (x, y)
            self.grid[x][y] = "you"
            for row in self.grid:
                print(row)
            return True
        elif re.search("go to [(]?\d+ ?, ?\d+", command):
            arr = re.split("[^a-zA-Z0-9]", command) # guaranteed to contain 2 numbers
            x, y = self.extractCoord(arr)
            if len(self.grid) == 0 or len(self.grid[0]) == 0:
                print("grid is not initialized yet")
                return False
            if x < 0 or x >= len(self.grid) or y < 0 or y >= len(self.grid[0]):
                print("coordinate is out of grid")
                return False
            if self.pos == (-1, -1):
                print("current position hasn't been initialized")
                return False
            target = (x, y)
            self.grid[x][y] = "target"
            for row in self.grid:
                print(row)
            if target == self.pos:
                print("you are already there")
                return True
            G = Graph(self.grid)
            moves = A_star(G, self.pos, target, manhattan_distance_heuristic)
            if moves is None:
                print("impossible to get to the target")
                return False
            else:
                print("**********************************************************************")
                print(moves)
                print("**********************************************************************")
            init_x, init_y = self.pos
            for i in range(1, len(moves)):
                if moves[i][1] > moves[i - 1][1]:
                    print("right")
                elif moves[i][1] < moves[i - 1][1]:
                    print("left")
                elif moves[i][0] > moves[i - 1][0]:
                    print("down")
                elif moves[i][0] < moves[i - 1][0]:
                    print("up")
                self.pos = moves[i]
            self.grid[x][y] = "you"
            self.grid[init_x][init_y] = ""
            for row in self.grid:
                print(row)
            return True
        return False

robot = Robot()
while True:
    command = input("You: ").lower()
    if command == "quit":
        break
    else:
        robot.gridParser(command)