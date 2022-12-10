"""
https://www.codewars.com/kata/5b817c2a0ce070ace8002be0/train/python
"""

def display_board(board, width):
    output = ''
    separator = (width*3)+(width-1)

    for i in range(len(board)):
        output += f' {board[i]} '
        if i+1 < len(board):
            if (i+1) % width == 0:
                output += '\n' + ('-'*separator) + '\n'
            else:
                output += '|'
    return output
