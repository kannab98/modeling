from json import load, dump
from surface import kernel_default, kernel_cwm, Surface
with open("rc.json", "r", encoding='utf-8') as f:
    const = load(f)

grid_size = const["surface"]["gridSize"][0] 
c = const["constants"]["lightSpeed"][0]
z0 = const["antenna"]["z"][0]

kernels = const["surface"]["kernels"][0]
kernels = [ eval(kernel) for kernel in kernels ]

# surface = Surface(const)
# host_constants = surface.export()



