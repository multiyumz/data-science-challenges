# pylint: disable=missing-docstring

def sudoku_validator(grid):
    #edge cases
    if not grid:
        return False

    #rows by rows checking
    hset = set()
    for i in range(9):
        for j in range(9):
            if grid[i][j] in hset:
                return False
            else:
                hset.add(grid[i][j])
        hset = set()

    #cols by cols checking
    hset = set()
    for i in range(9):
        for j in range(9):
            if grid[j][i] in hset:
                return False
            else:
                hset.add(grid[j][i])
        hset = set()


    #3 by 3 check
    subs = [range(0,3), range(3,6), range(6,9)]
    subgrids = []
    for x in subs:
        for y in subs:
            subgrids.append([x,y])

    for (row_range, column_range) in subgrids:
        hset = set()
        for i in row_range:
            for j in column_range:
                if grid[i][j] in hset:
                    return False
                else:
                    hset.add(grid[i][j])

    return True
