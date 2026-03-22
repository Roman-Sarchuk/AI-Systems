from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional
from collections import deque
import time


START = [
    [7, 8, 3],
    [4, 5, 6],
    [1, 2, 0]
]

GOAL = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
]

DEPTH_LEVEL = 10


class Direction(Enum):
    DOWN = (1, 0)
    UP = (-1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)


@dataclass(frozen=True)
class Field:
    size: int
    tiles: Tuple[Tuple[int, ...], ...]
    empty_pos: Tuple[int, int]

    @staticmethod
    def from_list(field: list[list[int]]) -> "Field":
        size = len(field)
        tiles = tuple(tuple(row) for row in field)

        for i in range(size):
            for j in range(size):
                if tiles[i][j] == 0:
                    return Field(size, tiles, (i, j))

        raise ValueError("No empty cell (0) found")

    def move(self, direction: Direction) -> Optional["Field"]:
        dx, dy = direction.value
        x, y = self.empty_pos
        new_x, new_y = x + dx, y + dy

        if not (0 <= new_x < self.size and 0 <= new_y < self.size):
            return None

        new_tiles = [list(row) for row in self.tiles]
        new_tiles[x][y], new_tiles[new_x][new_y] = \
            new_tiles[new_x][new_y], new_tiles[x][y]

        return Field.from_list(new_tiles)
    
    def is_goal(self, goal: "Field") -> bool:
        for i in range(self.size):
            for j in range(self.size):
                if goal.tiles[i][j] == "*":
                    continue
                elif self.tiles[i][j] != goal.tiles[i][j]:
                    return False
        return True
    
    def __str__(self):
        return "\n".join(" ".join(str(cell) for cell in row) for row in self.tiles)


@dataclass
class Node:
    state: Field
    parent: Optional["Node"]
    move: Optional[Direction]


visited_state_count = 0
expanded_node_count = 0
total_generated_node_count = 0
max_queue_size = 1

def bfs(start: Field, goal: Field) -> Optional[Node]:
    global visited_state_count, expanded_node_count, total_generated_node_count, max_queue_size
    queue = deque([Node(start, None, None)])
    visited = {start}

    while queue:
        current_node = queue.popleft()
        current_state = current_node.state
        expanded_node_count += 1

        if current_state.is_goal(goal):
            visited_state_count = len(visited)
            return current_node

        for direction in Direction:
            new_state = current_state.move(direction)

            if new_state is None:
                continue
            
            total_generated_node_count += 1

            if new_state in visited:
                continue

            visited.add(new_state)
            queue.append(Node(new_state, current_node, direction))
            max_queue_size = max(max_queue_size, len(queue))

    visited_state_count = len(visited)
    return None


def dfs(start: Field, goal: Field) -> Optional[Node]:
    global visited_state_count, expanded_node_count, total_generated_node_count, max_queue_size

    stack = [Node(start, None, None)]
    visited = {start}

    while stack:
        current_node = stack.pop()
        current_state = current_node.state
        expanded_node_count += 1

        if current_state.is_goal(goal):
            visited_state_count = len(visited)
            return current_node

        for direction in Direction:
            new_state = current_state.move(direction)

            if new_state is None:
                continue

            total_generated_node_count += 1

            if new_state in visited:
                continue

            visited.add(new_state)
            stack.append(Node(new_state, current_node, direction))
            max_queue_size = max(max_queue_size, len(stack))

    visited_state_count = len(visited)
    return None



def reconstruct_path(node: Node):
    moves = []
    while node.parent is not None:
        moves.append(node.move)
        node = node.parent
    moves.reverse()
    return moves


def main():
    start_field = Field.from_list(START)
    goal_field = Field.from_list(GOAL)

    print("--- Task statement ---")
    print("Start state:")
    print(start_field)
    print("\nGoal state:")
    for i in range(goal_field.size):
        row = []
        for j in range(goal_field.size):
            cell = goal_field.tiles[i][j]
            row.append(str(cell) if cell != "*" else "*")
        print(" ".join(row))
    print("-----------------------")


    print("What to use?\n1. BFS\n2. DFS")
    choose = input("Enter your choice: ")

    result = None

    start_time = time.perf_counter()
    if choose == "1":
        result = bfs(start_field, goal_field)
    elif choose == "2":
        result = dfs(start_field, goal_field)
    else:
        print("Invalid choice.")
        return
    end_time = time.perf_counter()

    print("\n--- Performance Metrics ---")
    print("Execution time:", end_time - start_time, "s")
    print("Total generated nodes:", total_generated_node_count)
    print("Visited states:", visited_state_count)
    print("Expanded nodes:", expanded_node_count)
    print("Branching factor:", total_generated_node_count / expanded_node_count if expanded_node_count > 0 else 0)
    print("Max queue size:", max_queue_size)
    print("---------------------------")
    print()

    print("--- Results ---")
    if result:
        path = reconstruct_path(result)
        print("Solution found!")
        print("Moves:", [move.name for move in path])
        print("Number of moves:", len(path))

        print("\nFinal state:")
        print(result.state)

        print(f"\nDo you want to see the path of states (DEATH_LEVEL={DEPTH_LEVEL})? (y/n)")

        choice = input("Enter your choice: ")
        if choice.lower() == "y":
            current = result
            for _ in range(DEPTH_LEVEL):
                if current is None:
                    break
                print(current.state)
                print()
                if current.move is not None:
                    print(f"Move: {current.move.name}")
                    print()
                current = current.parent

    else:
        print("No solution found.")
    print("----------------")


if __name__ == "__main__":
    main()