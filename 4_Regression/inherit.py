#class Animal(object):
#    def __init__(self, name):
#        self.name = name
#    def greet(self):
#        print('Hello, I am ',self.name)
#
#class Dog(Animal):
#    def greet(self):
#        super(Dog, self).greet()   # Python3 可使用 super().greet()
#        print('WangWang...')
#
#dog = Dog('dog')
#dog.greet()

class Base(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        print('a+b = ', a+b)
    
    def multi(self, a, b):
        print('a*b = ', a*b)

class A(Base):
    def __init__(self, a, b, c):
        super(A, self).__init__(a, b)  # Python3 可使用 super().__init__(a, b)
        super(A, self).multi(b, c)
        self.c = c

A(1,2,10)
