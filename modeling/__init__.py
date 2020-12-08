from json import load
import collections

# Замена дефолтному property
class dispatcher(object):

    def __init__(self, attr):
        self.attr = attr
        self.__call = []
    
    @property 
    def call(self):
        return self.__call

    def __get__(self, obj, objtype):
        return getattr(obj, self.attr)

    def __set__(self, obj, value):
        print(self)
        print(obj)
        for func in self.call:
            if type(func).__name__ == "function":
                func(value)

        setattr(obj, self.attr, value )

class rcParams():
    with open("rc.json") as f:
        __rc__ = load(f)

    def __init__(self, **kwargs):
        self.__json2object__("rc.json")
        self.callbacks = {}
        self.callbacks = {}


    def __json2object__(self, file):

#     """
#     Преобразование полей конфигурационного файла rc.json 
#     в объекты класса и присвоение им соответствующих значений


#     Файл из json вида:

#         >> { ... "swell": { ... "speed": [ 10, ... ] }, ... }

#     преобразуется в переменную __rc__ типа dict, поля выделенные под размерности и комментарии отбрасываются):

#         >> __rc__["swell"] = { ... , "speed": 10 } 

#     после словарь __rc__ становится объектом класса:

#         >> rc.swell.speed
#         >> out: 10
#     """
        with open(file) as f:
            __rc__ = load(f)



        for Key, Value in __rc__.items():
            setattr(self, Key, type('rc', (object,), {}))
            attr = getattr(self, Key)
            setattr(attr, "call", {})
            for key, value in Value.items():
                setattr(attr, key, value[0])







        # return rc

# Кастомный list
# class alist(collections.UserList):
#     def __init__(self, *args, **kwargs):
#         data = kwargs.pop('data')
#         super().__init__(self, *args, **kwargs)
#         self.data = data

#     def append(self, item):
# #         print('No appending allowed.')
#         return self.data.append(item)



# def getset(name, getting, setting):

#     return property(lambda self: getting(getattr(self, name)),
#                     lambda self, val: setattr(self, name, setting(val)))
name = "wind"
val = 10
class Foo(object):                                   

    def __init__(self):               
        self._wind = 15



    
# value.call.append(lambda x: x)
# d = Foo()
# print(d.wind)
#     return None

# d.fset = fset
# d.value = 15
# 
# 

# config  = kwargs["config"] if "config" in kwargs else os.path.join(os.path.abspath(os.getcwd()), "rc.json")

rc = rcParams()
 
from .surface import Surface
from .spectrum import Spectrum
surface = Surface()
spectrum = Spectrum()