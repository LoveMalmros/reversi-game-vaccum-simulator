import numpy as np
import collections


class Robot:

    def __init__(self, row, column):
        self.field = [[0 for x in range(0, row)] for y in range(0, column)]
        self.states = row*column
        self.row = row
        self.column = column
        self.header = 0
        self.field[np.random.choice(np.arange(0, self.row))][np.random.choice(np.arange(0, self.column))] = 1
        self.check_header()
        
    def check_header(self):
        posx = self.return_pos()[0]
        posy = self.return_pos()[1]

        headerCheck = 0

        if posx == 0 and posy == 0:
            self.header = np.random.choice(np.arange(0, 4), p=[0, 1/2, 1/2, 0])
            headerCheck = 1

        elif posx == 0 and posy == self.column-1:
            self.header = np.random.choice(np.arange(0,4), p=[0, 0, 1/2, 1/2])
            headerCheck = 1

        elif posx == self.row-1 and posy == 0:
            self.header = np.random.choice(np.arange(0, 4), p=[1/2, 1/2, 0, 0])
            headerCheck = 1

        elif posx == self.row-1 and posy == self.column-1:
            self.header = np.random.choice(np.arange(0, 4), p =[1/2, 0, 0, 1/2])
            headerCheck = 1

        if headerCheck == 0:
            if posy == 0:
                self.header = np.random.choice(np.arange(0,4), p =[1/3, 1/3, 1/3, 0])
                headerCheck = 2

            elif posy == self.column-1:
                self.header = np.random.choice(np.arange(0, 4), p=[1/3, 0, 1/3, 1/3])
                headerCheck = 2

            elif posx == 0:
                self.header = np.random.choice(np.arange(0, 4), p =[0, 1/3, 1/3, 1/3])
                headerCheck = 2

            elif posx == self.column-1:
                self.header = np.random.choice(np.arange(0, 4), p =[1/3, 1/3, 0, 1/3])
                headerCheck = 2
        if headerCheck == 0:
            if self.header == 0:
                self.header = np.random.choice(np.arange(0, 4), p=[.7, .1, .1, .1])
            elif self.header == 1:
                self.header = np.random.choice(np.arange(0, 4), p=[.1, .7, .1, .1])
            elif self.header == 2:
                self.header = np.random.choice(np.arange(0, 4), p=[.1, .1, .7, .1])
            elif self.header == 3:
                self.header = np.random.choice(np.arange(0, 4), p=[.1, .1, .1, .7])


    def return_pos(self):
        for x in range(0, self.row):
            for y in range(0, self.column):
                if self.field[x][y] == 1:
                    return x, y, self.header

    def move(self):
        posx = self.return_pos()[0]
        posy = self.return_pos()[1]
        self.check_header()

        if self.header == 0:
            self.field[posx][posy] = 0
            self.field[posx-1][posy] = 1
        elif self.header == 1:
            self.field[posx][posy] = 0
            self.field[posx][posy+1] = 1
        elif self.header == 2:
            self.field[posx][posy] = 0
            self.field[posx+1][posy] = 1
        elif self.header == 3:
            self.field[posx][posy] = 0
            self.field[posx][posy-1] = 1
        




