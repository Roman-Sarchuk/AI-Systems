from enum import Enum
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


class Direction(Enum):
    UP = "U"
    DOWN = "D"
    LEFT = "L"
    RIGHT = "R"


class Field:
    """
    Field is a square matrix of integers. Each cell can be either 0 (empty cell) or number from 1 to 8.
    """

    def __init__(self, field: list[list[int]]):
        self._field: list[list[int]] = list()
        self._size: int = 0
        self._empty_cell_pos: tuple[int, int] = tuple()

        self.set_field(field)
    
    def __eq__(self, other: Field) -> bool:
        if not isinstance(other, Field):
            raise ValueError("Other object must be of type Field")
        return self._field == other._field
    
    def set_field(self, field: list[list[int]]):
        """
        Field is a square matrix of integers. Each cell can be either 0 (empty cell) or number from 1 to 8.
        """
        if len(field) != len(field[0]):
            raise ValueError("Field must be square matrix")
        self._size = len(field)

        self._field = field
        for k in range(self._size*self._size):
            j = k % self._size
            i = k // self._size
            if field[i][j] == 0:
                self._empty_cell_pos = (i, j)
                break
    
    def get_size(self) -> int:
        return self._size
    
    def get_empty_cell_pos(self) -> tuple[int, int]:
        return self._empty_cell_pos
    
    def get_cell_value(self, row, col) -> int:
        if row < 0 or row >= self._size or col < 0 or col >= self._size:
            return -1
        return self._field[row][col]
    
    def get_relative_cell_value(self, direction: Direction) -> int:
        row, col = self._empty_cell_pos
        if direction == Direction.UP:
            row -= 1
        elif direction == Direction.DOWN:
            row += 1
        elif direction == Direction.LEFT:
            col -= 1
        elif direction == Direction.RIGHT:
            col += 1
        else:
            raise ValueError("Invalid direction value")
        return self.get_cell_value(row, col)
    
    def swap_cells(self, pos1: tuple[int, int], pos2: tuple[int, int]):
        if pos1[0] < 0 or pos1[0] >= self._size or pos1[1] < 0 or pos1[1] >= self._size:
            raise ValueError("Position 1 is out of field bounds")
        
        if self._field[pos1[0]][pos1[1]] == 0:
            self._empty_cell_pos = pos2
        elif self._field[pos2[0]][pos2[1]] == 0:
            self._empty_cell_pos = pos1

        self._field[pos1[0]][pos1[1]], self._field[pos2[0]][pos2[1]] = \
            self._field[pos2[0]][pos2[1]], self._field[pos1[0]][pos1[1]]
        

class PuzzleController:
    """
    Field is a square matrix of integers. Each cell can be either 0 (empty cell) or number from 1 to 8.
    Win pattern is a dictionary that maps direction to cell value. 
    For example, if win pattern is {Direction.UP: 1, Direction.DOWN: 7, Direction.LEFT: 5, Direction.RIGHT: 6}, 
    then player wins if empty cell has 1 above it, 7 below it, 5 to the left of it and 6 to the right of it.
    """

    def __init__(self, field: list[list[int]], win_pattern: list[list]):
        self._field: Field = Field(field)
        self._field.set_field(field)

        self._win_pattern = list()

        self.set_win_pattern(win_pattern)
    
    def set_win_pattern(self, win_pattern: list[list]):
        if not self._field or len(win_pattern) != self._size:
            raise ValueError("Win pattern must be square matrix of the same size as field")
        
        self._win_pattern = win_pattern
    
    def try_move(self, direction: Direction) -> bool:
        """
        Try to move empty cell in given direction. If move is possible, move empty cell and return True. Otherwise, return False.
        """
        size = self._field.get_size()
        row, col = self._field.get_empty_cell_pos()

        if direction == Direction.UP:
            row -= 1
        elif direction == Direction.DOWN:
            row += 1
        elif direction == Direction.LEFT:
            col -= 1
        elif direction == Direction.RIGHT:
            col += 1
        else:
            raise ValueError("Invalid direction value")

        if row < 0 or row >= size or col < 0 or col >= size:
            return False

        self._field.swap_cells((row, col), self._field.get_empty_cell_pos())
        return True
    
    def is_win(self) -> bool:
        """
        Check if current field state is win pattern.
        """
        return self._field.get_relative_cell_value(Direction.UP) == self._win_pattern[Direction.UP] and \
               self._field.get_relative_cell_value(Direction.DOWN) == self._win_pattern[Direction.DOWN] and \
               self._field.get_relative_cell_value(Direction.LEFT) == self._win_pattern[Direction.LEFT] and \
               self._field.get_relative_cell_value(Direction.RIGHT) == self._win_pattern[Direction.RIGHT]
    
    
@dataclass
class Node:
    value: Any
    parent: Node | None = None
    children: list[Node] = field(default_factory=list)

    def __post_init__(self):
        if self.parent:
            self.parent.children.append(self)


START_FIELD = [
    [1, 2, 3],
    [5, 6, 0],
    [7, 8, 4]
]

WIN_PATTERN = {
    Direction.UP: 1,
    Direction.DOWN: 7,
    Direction.LEFT: 5,
    Direction.RIGHT: 6
}


def main():
    controller = PuzzleController(START_FIELD, WIN_PATTERN)


if __name__ == "__main__":
    main()