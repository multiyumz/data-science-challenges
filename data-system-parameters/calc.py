# pylint: disable=missing-module-docstring,missing-function-docstring,eval-used
import sys


def main():
    """Implement the calculator"""
    left_operand = int(sys.argv[1])
    opt_name = sys.argv[2]
    right_operand = int(sys.argv[3])

    if opt_name == "+":
        return left_operand + right_operand
    if opt_name == "-":
        return left_operand - right_operand
    if opt_name == "*":
        return left_operand * right_operand
    if opt_name == '/':
        return left_operand / right_operand


if __name__ == "__main__":
    print(main())
