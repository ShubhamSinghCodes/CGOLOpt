import tkinter
import pygame
from pygame.locals import *
from ctypes import windll, Structure, c_long, byref
import numpy as np
import clipboard
import sys


def getinput():
    try:
        droppedFile = sys.argv[1]
        with open(droppedFile) as f:
            data = f.read().strip()
    except IndexError:
        fileloc = input("Enter fileloc: ")
        with open(fileloc) as f:
            data = f.read().strip()
    data = data.split("\n")
    h, w, rep, l = data[0].split(",")
    h, w, rep, l = int(h), int(w), int(rep), int(l)
    data = data[1:]
    data = [[int(val) for val in item.split(",")] for item in data]
    return h, w, rep, l, data

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
                   enumerate(data[count])]
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
        if col == 1:
            self.col = 1 - self.col
            self.surf.fill((255 * self.col, 255 * self.col, 255 * self.col))
            self.screen.blit(self.surf, (self.x, self.y))

    def setreal(self, col):
        if self.col != col:
            self.col = col
            self.surf.fill((255 * col, 255 * col, 255 * col))
            self.screen.blit(self.surf, (self.x, self.y))

def copystate():
    state = ""
    statedata = [[displaydata[i][j].col for j, px in enumerate(py)] for i, py in enumerate(displaydata)]
    for row in statedata:
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


height, width, repline, n, data = getinput()
x = np.full(n, 0)
for lo, hi in data:
    x[lo:hi] = 1
x = x.reshape((int(n/(height*width)),height,width))
data = x.tolist()
count = 0
screen = get_display_size(height, width)
[[displaydata[i][j].set(int(px)) for j, px in enumerate(py)] for i, py in enumerate(data[count])]
pygame.display.update()
repldata = data[1] if repline == 0 else None

while True:
    events = pygame.event.get()
    for event in events:
        if event.type == QUIT:
            quit()
        elif event.type == MOUSEBUTTONDOWN and event.button == BUTTON_RIGHT:
            copystate()
        elif event.type == KEYDOWN and event.key == K_LEFT and count>0:
            [[displaydata[i][j].set(int(px)) for j, px in enumerate(py)] for i, py in enumerate(data[count])]
            pygame.display.update()
            count -= 1
        elif (event.type == KEYDOWN and event.key == K_RIGHT) or (pygame.key.get_pressed()[K_SPACE]):
            count += 1
            if count>=len(data):
                [[displaydata[i][j].setreal(int(px)) for j, px in enumerate(py)] for i, py in enumerate(repldata)]
                count = repline
                print("repeating")
            [[displaydata[i][j].set(int(px)) for j, px in enumerate(py)] for i, py in enumerate(data[count])]
            pygame.display.update()
        if repldata == None and count == repline-1:
            repldata = [[displaydata[i][j].col for j, px in enumerate(py)] for i, py in enumerate(displaydata)]