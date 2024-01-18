## Shape Challenge 📐

Welcome to the Shape Challenge! Let's dive into the exciting world of shapes! 🎉

### Objective 🎯

Your goal is to create a class hierarchy for different shapes. We have three classes: Shape, Rectangle, and Circle. Your task is to implement these classes and their respective methods.

### Classes and Methods 📝

#### Shape Class 🟨

The Shape class is the base class for all shapes and has the following attributes and methods:

Attributes:
- `name` 📛: Represents the name of the shape.
- `color` 🌈: Represents the color of the shape.

Methods:
- `__init__(self, color, name)`: Initializes the Shape object with the given name and color.
- `say_name(self)`: Prints the name of the shape. 🗣️


#### Rectangle Class 🟦

The Rectangle class inherits from the Shape class and adds functionality specific to rectangles:

Attributes:
- `width` ↔️: Represents the width of the rectangle.
- `height` ⬆️: Represents the height of the rectangle.

Methods:
- `__init__(self, color, name,width, height)`: Initializes the Rectangle object with the given name, color, width, and height.
- `say_name(self)`: Overrides the base class method to print the name of the rectangle along with its shape type ((e.g. "My name is Rei and I am a rectangle")). 🗣️🟦
- `area(self)`: Calculates and returns the area of the rectangle. 📐
- `perimeter(self)`: Calculates and returns the perimeter of the rectangle. 📏

#### Circle Class 🟣

The Circle class inherits from the Shape class and adds functionality specific to circles:

Attributes:
- `radius` ⚪: Represents the radius of the circle.

Methods:
- `__init__(self, color, name, radius)`: Initializes the Circle object with the given name, color, and radius.
- `say_name(self)`: Overrides the base class method to print the name of the circle along with its shape type (e.g. "My name is Kvothe and I am a circle"). 🗣️🟣
- `area(self)`: Calculates and returns the area of the circle. 📐
- `perimeter(self)`: Calculates and returns the perimeter of the circle. 📏

### Testing ✅

To test your implementation, you can run the provided tests in the `tests` directory. Execute the following command in your terminal:

```bash
make pytest
