from sympy import symbols
from sympy.geometry import Point, Triangle, intersection, Polygon


def polygon():
    p1, p2, p3, p4, p5 = [(0, 0.5), (1, 0), (5, 1), (0, 1), (3, 0)]
    p = Polygon(p1, p2, p3, p4, p5)
    print(p)


polygon()
