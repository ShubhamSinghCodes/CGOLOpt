import tkinter
import pygame
from pygame.locals import *
from ctypes import windll, Structure, c_long, byref
import numpy as np
from scipy.signal import convolve2d as convolve
import sys


def getinput():
    global r,T
    T=10
    r = 3
    try:
        droppedFile = "seeds/LeniaStuff/Bug (bigger).txt"#sys.argv[1]
        global info
        with open(droppedFile) as f:
            data = f.read().strip()
            data = [item.strip() for item in data.split("\n")]
            info = data[0].split(",")
            info[0] = int(info[0])
            info[3] = [float(val) for val in info[3].split("|")]
            data = data[1:]
            data = [[float(cell) if cell!="" else 0.0 for cell in item.split(",")] for item in data]
            mlen = max([len(row) for row in data])
            [row.extend([0.0]*(mlen-len(row))) for row in data]
            data = np.array(data)
        return data
    except IndexError:
        pass
    r = input("enter radius: ")
    r = 3 if r=="" else int(r)
    dim = input("enter dimensions: ")
    dim = 100 if dim=="" else int(dim)
    w = h = dim*r
    T = input("enter T: ")
    T = 10 if T=="" else int(T)
    grid = np.random.random((h,w))
    grid = np.multiply(np.random.randint(0,2,(h,w)),grid)
    return grid

def get_display_size(h,w):
    global displaydata,r
    root = tkinter.Tk()
    root.update_idletasks()
    root.attributes('-fullscreen', True)
    root.state('iconic')
    height = root.winfo_screenheight()
    width = root.winfo_screenwidth()
    root.destroy()
    height -= 100
    sqsize = min(height/h,width/w)
    screen = pygame.display.set_mode((int(sqsize * w), int(sqsize * h)))
    displaydata = [[Square(sqsize * i, j * sqsize, sqsize + 1, screen) for j, px in enumerate(py)] for i, py in
                   enumerate(celldata)]
    class RECT(Structure):
        _fields_ = [
            ('left', c_long),
            ('top', c_long),
            ('right', c_long),
            ('bottom', c_long),
        ]

        def width(self):  return self.right - self.left

        def height(self): return self.bottom - self.top

    SetWindowPos = windll.user32.SetWindowPos
    GetWindowRect = windll.user32.GetWindowRect
    rc = RECT()
    GetWindowRect(pygame.display.get_wm_info()['window'], byref(rc))
    SetWindowPos(pygame.display.get_wm_info()['window'], -1, rc.left, rc.top, 0, 0, 0x0001)
    del rc
    return sqsize, screen, r

class Square(pygame.sprite.Sprite):
    def __init__(self, y, x, size, screen):
        super(Square, self).__init__()
        self.surf = pygame.Surface((size, size))
        self.rect = self.surf.get_rect()
        self.screen = screen
        self.screen.blit(self.surf, (x, y))
        self.x = x
        self.y = y
        self.col = 0

    def set(self, col):
            self.col = col
            self.surf.fill((255 * col, 255 * col, 255 * col))
            self.screen.blit(self.surf, (self.x, self.y))


bell = lambda x, m, s: np.exp(-((x-m)/s)**2 / 2)
info = ""
celldata = getinput()
display = True
_, screen, r = get_display_size(celldata.shape[0],celldata.shape[1])
if type(info)==list:
    r,kernel_type,rule,params=info
else:
    kernel_type = input("Enter kernel type(1s,pow<n>,exp,euclid,1scircle,pow<n>circle,expcircle,euclidcircle,bell): ")
    kernel_type = "1scircle" if kernel_type=="" else kernel_type.strip()
    rule = input("Enter rule(cgol,smoothlife,lenia): ")
    rule = "lenia" if rule == "" else rule.strip().lower()
if kernel_type.startswith("1s"):
    conv_kernel = np.ones((2*r+1,2*r+1))
    conv_kernel[r][r] = 0
elif kernel_type.startswith("pow") and kernel_type.endswith("circle"):
    p = int(kernel_type[3:-6])
    conv_kernel = np.array([[100]])
    for i in range(r):
        conv_kernel = np.pad(conv_kernel, 1, 'constant', constant_values=(p**(i), p**(i)))
    conv_kernel = 1/conv_kernel
    conv_kernel[r][r]=0
elif kernel_type.startswith("pow"):
    p = int(kernel_type[3:])
    conv_kernel = np.array([[100]])
    for i in range(r):
        conv_kernel = np.pad(conv_kernel, 1, 'constant', constant_values=(p**(i), p**(i)))
    conv_kernel = 1/conv_kernel
    conv_kernel[r][r]=0
elif kernel_type.startswith("exp"):
    conv_kernel = np.array([[100]])
    for i in range(r):
        conv_kernel = np.pad(conv_kernel, 1, 'constant', constant_values=(i,i))
    conv_kernel = -conv_kernel
    conv_kernel = np.exp(conv_kernel)
    conv_kernel[r][r]=0
elif kernel_type.startswith("euclid"):
    xarr = np.array([range(2*r+1)]*(2*r+1)) - r
    yarr = np.array([[item]*(2*r+1) for item in range(2*r+1)]) - r
    conv_kernel = np.sqrt(np.add(np.square(xarr),np.square(yarr)))
    conv_kernel = 1/conv_kernel
    conv_kernel[r][r]=0
elif kernel_type=="bell":
    D = np.linalg.norm(np.asarray(np.ogrid[-r:r, -r:r]) + 1) / r
    K = (D < 1) * bell(D, 0.5, 0.15)
    K = K / np.sum(K)
    conv_kernel = K
elif kernel_type.startswith("leniasave"):
    kernel_core = {
        # [0,1] -> [0,1]
        1: lambda r: (r > 0) * (r < 1) * (4 * r * (1 - r)) ** 4,  # polynomial (quad4)
        2: lambda r: (r > 0) * (r < 1) * np.nan_to_num(np.exp(4 - 1 / (r * (1 - r))), 0),
        # exponential / gaussian bump (bump4)
        3: lambda r, q=1 / 4: (r >= q) * (r <= 1 - q),  # step (stpz1/4)
        4: lambda r: (r > 0) * (r < 1) * np.exp(- ((r - 0.5) / 0.15) ** 2 / 2)  # exponential / leaky gaussian bump
    }
    I, J = np.meshgrid(np.arange(r), np.arange(r))
    X = (I - int(r/2)) / r
    Y = (J - int(r/2)) / r
    D = np.sqrt(X ** 2 + Y ** 2)
    def kernel_shell(r):
        bs = np.array([1.0])
        b = bs[np.minimum(np.floor(r).astype(int), 0)]
        kfunc = kernel_core[int(kernel_type[9])]
        return (r < 1) * kfunc(np.minimum(r % 1, 1)) * b

    kernel = kernel_shell(D)
    print(kernel)
    conv_kernel = kernel
if kernel_type.endswith("circle"):
    xarr = np.array([range(2 * r + 1)] * (2 * r + 1)) - r
    yarr = np.array([[item] * (2 * r + 1) for item in range(2 * r + 1)]) - r
    circle_kernel = np.sqrt(np.add(np.square(xarr), np.square(yarr)))
    circle_kernel = 1*(circle_kernel<=circle_kernel[0,r])
    conv_kernel = np.multiply(conv_kernel,circle_kernel)
conv_kernel = conv_kernel / np.sum(conv_kernel)
if rule=="cgol":
    T = 1
    def actual(n):
        n = np.round(n) * 8
        return 0 + (n==3) - ((n<2)|(n>3))
elif rule=="smoothlife":
    if type(info) == list:
        global b1,b2,s1,s2
        b1,b2,s1,s2,T = params
    else:
        for var,default in [("b1",2.5/8),("b2",3.5/8),("s1",1.5/8),("s2",3.5/8)]:
            globals()[var] = input(f"enter {var}: ")
            globals()[var] = default if globals()[var]=="" else float(globals()[var])
    def actual(n):
        return 0 + ((n>=b1)&(n<=b2)) - ((n<s1)|(n>s2))
elif rule=="lenia":
    if type(info) == list:
        global m,s
        m,s,T = params
    else:
        for var,default in [("m",0.29),("s",0.035)]:
            globals()[var] = input(f"enter {var}: ")
            globals()[var] = default if globals()[var]=="" else float(globals()[var])
    def actual(n):
        return bell(n, m, s) * 2 - 1
elif rule.startswith("leniasave"):
    m,s,T = params
    growth_func = {
        # [0,1] -> [-1,1]
        1: lambda n, m, s: np.maximum(0, 1 - (n - m) ** 2 / (9 * s ** 2)) ** 4 * 2 - 1,  # polynomial (quad4)
        2: lambda n, m, s: np.exp(- (n - m) ** 2 / (2 * s ** 2)) * 2 - 1,  # exponential / gaussian (gaus)
        3: lambda n, m, s: (np.abs(n - m) <= s) * 2 - 1  # step (stpz)
    }
    gfunc = growth_func[int(rule[9])]
    def actual(n):
        return gfunc(n, m, s)
def step():
    global celldata, conv_kernel
    n = convolve(celldata,conv_kernel,boundary="wrap",mode="same")
    celldata = np.clip(np.add(celldata,actual(n)*(1/T)),0,1)


def displaydec(f):
    def wrapper():
        f()
        [[displaydata[i][j].set(px) for j, px in enumerate(py)] for i, py in enumerate(celldata)]
        pygame.display.update()
        events = pygame.event.get()
        [quit() for event in events if event.type == QUIT]
    return wrapper

if display:
    step = displaydec(step)

print("Started")
while True:
    step()

