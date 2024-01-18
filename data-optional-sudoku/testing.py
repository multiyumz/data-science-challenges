def testing_grid(grid):
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


grid = [
            [7,8,4,  1,5,9,  3,2,6],
            [5,3,9,  6,7,2,  8,4,1],
            [6,1,2,  4,3,8,  7,5,9],

            [9,2,8,  7,1,5,  4,6,3],
            [3,5,7,  8,4,6,  1,9,2],
            [4,6,1,  9,2,3,  5,8,7],

            [8,7,6,  3,9,4,  2,1,5],
            [2,4,3,  5,6,1,  9,7,8],
            [1,9,5,  2,8,7,  6,3,4]
        ]

subs = [range(0,3), range(3,6), range(6,9)]
subgrids = []
for x in subs:
    for y in subs:
        # print(x,y)
        subgrids.append([x,y])
print(subgrids)


# print(testing_grid(grid))
# print(subs)
