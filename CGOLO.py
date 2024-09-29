import tkinter
import pygame
from pygame.locals import *
from ctypes import windll, Structure, c_long, byref
import numpy as np
from scipy.signal import convolve2d as convolve
import clipboard
import sys


def getinput():
    try:
        droppedFile = sys.argv[1]
        with open(droppedFile) as f:
            data = f.read()
            data = data.replace("0",".").replace("1","O")
            data = [item.strip() for item in data.split("\n")]
            maxsize = max([len(item) for item in data])
            data = [item.ljust(maxsize,".") for item in data]
            data = [[int(cell) for cell in list(item.replace(".","0").replace("O","1"))] for item in data]
            data = np.array(data)
        return data
    except IndexError:
        pass
    filetype = input("Enter filetype: ")
    if filetype == "":
        h = input("enter height: ")
        h = 100 if h=="" else int(h)
        w = input("enter width: ")
        w = 100 if w=="" else int(w)
        return np.random.randint(0,2,(h,w))
    elif filetype == "seeds":
        from os import listdir
        for file in listdir(r"seeds"):
            if file.endswith(".txt"):
                print(file)
        fname = input("enter filename: ")
        with open(f"seeds/{fname}") as f:
            data = f.read()
            data = data.replace("0",".").replace("1","O")
            data = [item.strip() for item in data.split("\n")]
            maxsize = max([len(item) for item in data])
            data = [item.ljust(maxsize,".") for item in data]
            data = [[int(cell) for cell in list(item.replace(".","0").replace("O","1"))] for item in data]
            data = np.array(data)
        return data

def get_display_size(h,w):
    global displaydata
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
    return sqsize, screen

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
        if self.col != col:
            self.col = col
            self.surf.fill((255 * col, 255 * col, 255 * col))
            self.screen.blit(self.surf, (self.x, self.y))

def copystate():
    state = ""
    for row in celldata:
        for cell in row:
            state += "b" if cell == 0 else "o"
        state += "$"
    short = ""
    count = 0
    last = state[0]
    for i,char in enumerate(list(state)):
        if char == last:
            count += 1
        else:
            if count == 1:
                short += f"{last}"
            else:
                short += f"{count}{last}"
            count = 1
            last = char
    if count != 0:
        if count == 1:
            short += f"{last}"
        else:
            short += f"{count}{last}"
    clipboard.copy(short)


celldata = getinput()
display = input("Display? (nothing means yes, anything else means no): ") == ""
savedata = input("Save History? (nothing means no, anything else means yes): ") != ""
fps = input("Input target fps(nothing means max): ")
if not fps == "":
    fps = float(fps)
else:
    fps = 0
screen = get_display_size(celldata.shape[0],celldata.shape[1])
conv_kernel = np.array([[1,1,1],[1,0,1],[1,1,1]])

def step():
    global celldata, conv_kernel
    n = convolve(celldata, conv_kernel, boundary='wrap', mode='same')
    temp = n-3
    temp = temp * (temp + celldata)
    celldata = 1 * (temp == 0)
    [copystate() for event in pygame.event.get() if event.type == MOUSEBUTTONDOWN and event.button == BUTTON_RIGHT]

if savedata:
    def save():
        global saveobject
        x = np.array(saveobject.changes)
        where = np.flatnonzero
        n = len(x)
        starts = np.r_[0, where(~np.isclose(x[1:], x[:-1], equal_nan=True)) + 1]
        lengths = np.diff(np.r_[starts, n])
        values = x[starts]
        starts = starts[values==1]
        lengths = lengths[values==1]
        ends = starts + lengths
        fname = input('filename: ')
        with open(f"histories/{fname}.cgolhist","x"):
            pass
        with open(f"histories/{fname}.cgolhist","a") as f:
            f.write(f"{celldata.shape[0]},{celldata.shape[1]},{saveobject.repeatframe},{n}\n")
            for start,end in zip(starts,ends):
                f.write(f"{start},{end}\n")
        print("done")
        quit()

def displaydec(f):
    if savedata:
        def wrapper():
            f()
            [[displaydata[i][j].set(int(px)) for j, px in enumerate(py)] for i, py in enumerate(celldata)]
            pygame.display.update()
            events = pygame.event.get()
            [save() for event in events if event.type == QUIT]
    else:
        def wrapper():
            f()
            [[displaydata[i][j].set(int(px)) for j, px in enumerate(py)] for i, py in enumerate(celldata)]
            pygame.display.update()
            events = pygame.event.get()
            [quit() for event in events if event.type == QUIT]
    return wrapper
def fpsdec(f,fps,clock):
    def wrapper():
        f()
        clock.tick(fps)
    return wrapper
def savedec(f):
    global saveobject
    class Saver:
        def __init__(self,celldata):
            self.changes = celldata.flatten().tolist()
            self.frames = []
            self.repeatframe = -1

        pass
    saveobject = Saver(celldata)
    def wrapper():
        original = celldata.copy()
        f()
        shortened = ("".join(["".join([str(cell) for cell in row]) for row in celldata.tolist()]))
        if shortened in saveobject.frames:
            saveobject.repeatframe = saveobject.frames.index(shortened)
            save()
        else:
            change = 1 * (original != celldata)
            saveobject.changes.extend(change.flatten().tolist())
            saveobject.frames.append(shortened)
    return wrapper

if display:
    step = displaydec(step)
if fps != 0:
    clock = pygame.time.Clock()
    step = fpsdec(step,fps,clock)
if savedata:
    step = savedec(step)

while True:
    step()

