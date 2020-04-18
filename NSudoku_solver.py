"""
Sudoku Solver for Standard Rules, size NxN
"""
import math
import numpy as np
from pulp import *
from time import time


def gen_vertices(size, start_nums={}):
    """Generates a dictionary of all vertices, their values and their indices
    Args:
        size (int): Size of the Sudoku grid, since it must be symmetrical
        start_nums (dict): starting numbers in dict form { coord_tup : val }

    Output:
        dict : { index : [(x, y), c] }
                where index ranges from 1 to size ** 2, (x,y) is a coordinate tuple,
                c is the colour at that index
    """
    grid_range = range(1, size + 1)
    return {(size * (i - 1) + j): [(i, j), (str(start_nums[(i, j)]) if (i, j) in start_nums.keys() else 0)]
            for i in grid_range
            for j in grid_range}


def gen_edges(size):
    """Generates the set of all edges of the graph that corresponds to the sudoku grid.
    Edges are tuples of vertex indices: ( index_of(start_x, start_y), index_of(end_x, end_y) )
    """
    # setup values to determine the location on the board
    sub_grid = int(math.sqrt(size))
    grid_range = range(1, size + 1)
    quadrants = [[i for i in range(j, j + sub_grid)] for j in range(1, size + 1, sub_grid)]

    # vertices are indexed left to right row-wise down the grid to help with ILP later
    vertices = gen_vertices(size)
    index = {v[0]: k for k, v in vertices.items()}

    edges = set()
    for i, v in vertices.items():
        z = v[0]
        x_quad = next(filter(lambda q: z[0] in q, quadrants))
        y_quad = next(filter(lambda q: z[1] in q, quadrants))
        # Row edges (all colours used in row v[0])
        edges = edges.union({tuple(sorted([i, index[(z[0], j)]])) for j in grid_range})

        # Column edges (all colours used in column v[1])
        edges = edges.union({tuple(sorted([i, index[(j, z[1])]])) for j in grid_range})

        # Sub_grid edges (all colours are used in a subgrid that corresponds to x_quad, y_quad)
        edges = edges.union({tuple(sorted([i, index[(j, k)]])) for j in x_quad for k in y_quad})

        # Exclude edge that connects to self
        edges = edges.difference({(index[z], index[z])})
    return edges


def perfect_square(num):
    """Checks if the provided num (int) is a perfect square, returns bool"""
    root = math.sqrt(num)
    return int(root + 0.5) ** 2 == num


def display(grid, colours):
    """Displays the grid"""
    size = len(grid)
    sub_size = int(math.sqrt(size))
    empty_cell = '.'
    s = ''
    for row in range(size):
        row_display = ''
        if row in [sub_size * i for i in range(1, sub_size)]:
            s += ('-' * sub_size * 2 + ' + ')
            s += ('-' * (sub_size * 2 - 1) + ' + ') * (sub_size - 2)
            s += '-' * sub_size * 2 + "\n"
        for col in range(size):
            if col in [sub_size * i for i in range(1, sub_size)]:
                row_display = ' '.join([row_display, '|'])
            curr_index = grid[row][col]
            row_display = ' '.join([row_display, colours[curr_index] if curr_index > 0 else empty_cell])
        s += row_display + "\n"
    s += "\n"
    return s


def N_solve(size, start_nums, colours, solve_output=None):
    """Solve a general sudoku of size (int)
    Args:
        size (int): the size of the sudoku grid, equal to the number of squares in a row
                    Size must be a perfect square.
        start_nums (dict): A dictionary containing the starting numbers for sudoku puzzle
                           { coord (tuple): value (int) }
                            Example size = 9, START_NUMS = {
                                (1, 3): 1, (1, 4): 7, (1, 5): 2, (1, 6): 5,
                                (2, 2): 8, (2, 5): 1,
                                (3, 1): 2, (3, 2): 5, (3, 7): 1, (3, 8): 3,
                                (4, 2): 7, (4, 7): 5,
                                (5, 4): 1, (5, 5): 8, (5, 6): 6,
                                (6, 3): 9, (6, 8): 8,
                                (7, 2): 4, (7, 3): 5, (7, 8): 2, (7, 9): 9,
                                (8, 5): 9, (8, 8): 6,
                                (9, 4): 6, (9, 5): 4, (9, 6): 8, (9, 7): 3
                            }
        colours (iter): An iterable containing single digit strings of possible colors
                        Example, colours = list(map(str, range(1, GRID_SIZE + 1)))
        solve_output (str): string filepath to output file for solver
     """
    if not perfect_square(size):
        print("Invalid board size, must be a perfect square")
        return

    # create a colour dictionary
    ci = {j: i for i, j in enumerate(colours, start=1)}
    ci.update({0: 0})
    colours = {i: j for i, j in enumerate(colours, start=1)}
    if len(colours) != size:
        print("Number of colours not equal to board size")
        return

    # Run checks, prevent from running
    for x, y in start_nums.keys():
        if x > size or y > size:
            print("Incorrect starting grid squares")
            return

    # Generate the starting board
    start_board = np.zeros(shape=(size, size), dtype='int8')
    board_val = gen_vertices(size, start_nums)
    for k, v in board_val.items():
        z = v[0]
        c = v[1]
        start_board[z[0] - 1, z[1] - 1] = ci[c]

    # Initialise the sudoku model
    sudoku_model = LpProblem("Sudoku Problem", LpMinimize)

    # Colour variable and constraint
    var_colour = LpVariable.dicts('D', colours.keys(), cat=LpBinary)

    # Define objective function
    sudoku_model += sum(var_colour.values()) * 0

    # Define colour constraint
    # All colours must be used
    for c in colours.keys():
        sudoku_model += var_colour[c] == 1

    # Define constraints for vertices
    # Z_(v,c), v = all vertices in V, c = all possible colours
    # Z_(v,c) = 1 if vertex v has colour c, 0 otherwise (binary constraint)
    # Exactly one color can be assigned per vertex
    vertex_index = board_val.keys()
    var_vertices = LpVariable.dicts('Z', (vertex_index, colours.keys()), cat=LpBinary)
    for v in vertex_index:
        sudoku_model += sum(var_vertices[v][c] for c in colours.keys()) == 1

    # Define edge constraints
    # Z_(x,c) + Z_(y,c) <= D_c, for (x,y) in edges
    # Every vertex incident on the same edge must not share the same color for all c
    edges = gen_edges(size)
    for u, v in edges:
        for c in colours.keys():
            sudoku_model += sum((var_vertices[u][c], var_vertices[v][c])) <= 1

    # Define pre-fill constraints
    # Z_(z,c) == 1 for vertices z where the colour is c
    for z, l in board_val.items():
        if ci[l[1]] != 0:
            sudoku_model += var_vertices[z][ci[l[1]]] == 1

    # Solve the sudoku board
    start = time()
    sudoku_model.solve()
    end = time()

    # Write the sudoku solution
    solved_board = start_board.copy()
    for i, v in var_vertices.items():
        for c, val in v.items():
            if val.value() == 1:
                board_val[i][1] = colours[c]
    for k, v in board_val.items():
        z = v[0]
        c = v[1]
        solved_board[z[0] - 1, z[1] - 1] = ci[c]

    # Write the solver information
    solve_string = f"Solver status: {LpStatus[sudoku_model.status]}\n"
    solve_string += f"Solver time taken: {(end - start):.4f} seconds\n"
    solve_string += "The starting board is:\n"
    solve_string += display(start_board, colours)
    solve_string += "The solved board is:\n"
    solve_string += display(solved_board, colours)

    # Provide for output options
    if solve_output:
        with open(solve_output, 'w') as f:
            print(solve_string, file=f)
    else:
        print(solve_string)

    return board_val


if __name__ == '__main__':
    # (int): The size of the grid
    GRID_SIZE = 16

    # (dict): { coord_tup: int }
    # The existing numbers in the sudoku problem, input as follows
    # (x, y): starting_value
    START_NUMS = {
        (1, 2): "D", (1, 3): "E", (1, 6): "F", (1, 13): 6, (1, 15): "B", (1, 16): 9,
        (2, 1): 9, (2, 2): "A", (2, 3): "B",
        (3, 3): "F", (3, 4): "C", (3, 10): "D", (3, 14): "E",
        (4, 1): 8, (4, 3): 7, (4, 8): "C", (4, 9): "E", (4, 12): "B", (4, 13): 4, (4, 14): 5,
        (5, 2): "C", (5, 6): 8, (5, 8): 1, (5, 10): 3, (5, 12): 9, (5, 15): "E",
        (6, 2): "B", (6, 4): 1, (6, 6): "E", (6, 8): 9, (6, 10): 8, (6, 11): "F", (6, 14): 4, (6, 16): "D",
        (7, 7): "A", (7, 8): "B", (7, 11): 5, (7, 12): 7, (7, 14): 2,
        (8, 5): "C", (8, 7): "F", (8, 8): 3, (8, 9): 0, (8, 10): "A", (8, 16): "B",
        (9, 4): "A", (9, 6): 9, (9, 8): "F", (9, 10): 7, (9, 12): "C",
        (10, 1): 1, (10, 4): 2, (10, 5): 7, (10, 7): "B", (10, 9): 9, (10, 11): 4, (10, 12): "E",
        (11, 1): "E", (11, 5): 8, (11, 7): 4, (11, 9): 5, (11, 12): "A", (11, 16): "F",
        (12, 3): 3, (12, 4): 8, (12, 5): "D", (12, 6): 6, (12, 7): 1, (12, 8): "E", (12, 10): 0, (12, 13): 5, (12, 16): "A",
        (13, 1): 6, (13, 3): 8, (13, 4): "B", (13, 7): "C", (13, 8): 7, (13, 11): "D",
        (14, 2): 0, (14, 4): "E", (14, 5): 3, (14, 7): 8,
        (15, 1): "C", (15, 2): 4, (15, 5): "E", (15, 6): "A", (15, 7): 9, (15, 8): 6, (15, 9): 7, (15, 12): "F",
        (15, 15): 3,
        (16, 2): 7, (16, 7): 2, (16, 10): "B", (16, 11): 0, (16, 12): 5, (16, 14): "C", (16, 15): "D"
    }

    # (iter): List of all possible colourings, must be str
    COLOUR_VALS = list(map(str, [*range(10), *'ABCDEF']))

    # (str): Output filepath string
    OUTPUT = 'p1_8.txt'

    N_solve(GRID_SIZE, START_NUMS, COLOUR_VALS, OUTPUT)
