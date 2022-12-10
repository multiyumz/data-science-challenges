"""Dummy challenge for Kitt Demo"""



from signal import raise_signal
import math

def circle_area(radius):
    """Returns the area of the circle of given radius"""
    if radius > 0:
        return radius * radius * math.pi
    return 0
