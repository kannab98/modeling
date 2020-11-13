from json import load

class rcParams():

    def __init__(self, **kwargs):
        pass

    @staticmethod
    def __json2object__(file):

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

        rc = type('Config', (object,), {})()


        for Key, Value in __rc__.items():
            setattr(rc, Key, type('rc', (object,), {}))
            for key, value in Value.items():
                __rc__[Key][key] = value[0]
                attr = getattr(rc, Key)
                setattr(attr, key, value[0])
        
        return rc


# config  = kwargs["config"] if "config" in kwargs else os.path.join(os.path.abspath(os.getcwd()), "rc.json")
rc = rcParams.__json2object__("rc.json")