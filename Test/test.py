
class T:
    def __init__(self):
        self.a = None
        self.b = None
    def f(self):
        print(self.a)
    @property
    def params(self):
        return self.a, self.b

if __name__ == '__main__':
    t = T()
    a, b = t.params
    print(t.params)
    t.f()
