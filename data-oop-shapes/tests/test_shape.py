import unittest
from shape import Shape, Rectangle, Circle

class TestShape(unittest.TestCase):
    def test_shape_say_name(self):
        shape = Shape("red", "Shape")
        self.assertEqual(shape.say_name(), "My name is Shape")

    def test_rectangle_say_name(self):
        rectangle = Rectangle("blue", "Bob", 4, 5)
        self.assertEqual(rectangle.say_name(), "My name is Bob and I am a rectangle.")

    def test_rectangle_area(self):
        rectangle = Rectangle("red", "Rectangle", 4, 5)
        self.assertEqual(rectangle.area(), 20)

    def test_rectangle_perimeter(self):
        rectangle = Rectangle("red", "Rectangle", 4, 5)
        self.assertEqual(rectangle.perimeter(), 18)

    def test_circle_say_name(self):
        circle = Circle("green", "Dora", 3)
        self.assertEqual(circle.say_name(), "My name is Dora and I am a circle.")

    def test_circle_area(self):
        circle = Circle("blue", "Circle", 3)
        self.assertAlmostEqual(circle.area(), 28.26, places=1)

    def test_circle_perimeter(self):
        circle = Circle("blue", "Circle", 3)
        self.assertAlmostEqual(circle.perimeter(), 18.84, places=1)

class TestInheritance(unittest.TestCase):
    def test_rectangle_inheritance(self):
        self.assertTrue(issubclass(Rectangle, Shape), "Rectangle class should inherit from Shape")

    def test_circle_inheritance(self):
        self.assertTrue(issubclass(Circle, Shape), "Circle class should inherit from Shape")




if __name__ == '__main__':
    unittest.main()
