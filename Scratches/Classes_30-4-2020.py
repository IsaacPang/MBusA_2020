## Pointers
# What does pointing mean?
# What does the following code do?
a = [1, 2, 3]
b = a
b[0] = 2
print(f"a is {a}, b is {b}")

print(id(a))
print(id(b))

# How to avoid pointing?
# What does the following code do now?
a = [1, 2, 3]
b = [1, 2, 3]
b[0] = 2
print(f"a is {a}, b is {b}")

b = 1
a = [b]
print(a)

b = 2
print(a)


b = [1]
a = [b]
print(a)

b = [2]
print(a)

## Objects (Classes)
# What is an object?
# Objects can have attributes and specific methods
# Defining as Python objects
class Player:
    HAIR = "white"
    def __init__(self):
        self.skin = "brown"
        self.value = 2

print(Player)
print(Player())

print(Player.HAIR)

print(Player.skin)
print(Player().skin)

p1 = Player()
p2 = Player()
assert(Player == Player)
assert(p1 == p2)

print(id(p1))
print(id(p2))

print(p1.HAIR)
print(p2.HAIR)

print(p1.skin)
print(p2.skin)
p1.skin = "black"
print(p1.skin)
print(p2.skin)


p1.HAIR = "black"
print(p2.HAIR)
print(p1.HAIR)

id(Player)

Player.HAIR = "blond"
print(p1.HAIR)
print(p2.HAIR)

Player.skin = "yellow"
print(p1.skin)
print(p2.skin)

## Functions vs Class methods
# Functions are static
def my_func(x):
    return 2 * x

print(my_func(2))

# Class methods are 'dynamic'
class Functions:
    def __init__(self):
        self.value = 1
        self.anything = 5

    def my_func(self):
        return "This is a method"

    def add(self, x, y):
        return self.anything + x.value + y.value

    def addagain(self, x, y):
        return x + y

    def __str__(self):
        return "String Representation"

    def __repr__(self):
        return "Actual representation"


f = Functions()

str(f)
repr(f)
print(f)

p1 = Player(); p2 = Player()
f.my_func()
f.add(p1, p2)
f.addagain(1, 3)

class Parent:
    def __init__(self):
        self.job = 0


class Child(Parent):
    def __init__(self):
        super().__init__()
        self.skin = "black"

p = Parent(); c = Child()
