from tkinter import *
import numpy as np
import operator
import copy
import time
import robot
from random import randint


NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3


class OthelloGame:
    master = Tk()
    m = 4
    n = 4
    board = np.zeros((m, n))
    w = Canvas(master, width=600, height=600)
    robot = robot.Robot(4, 4)
    robot.init_robot()
    O = []

    def set_start_index(self):
        self.index = randint(0, 15)

    def return_callback(self, event):
        self.robot.move()
        self.render_grid()

    def render_grid(self):
        pos = self.robot.return_pos()
        # print(self.robot.sensor())
        for i in range(0, 4):
            for j in range(0, 4):
                self.w.create_rectangle(i*150, j*150, (i+1)*150,
                                        (j+1)*150, fill="white", outline='black')
        for i, val in enumerate(np.nditer(self.get_obs_matrix())):
            x = int(i % self.n)
            y = int(i / self.n)
            print(x, y)
            if val == 0.1:
                self.w.create_oval((x*150)+35, (y*150)+35, ((x+1)
                                                            * 150)-35, ((y+1)*150)-35, fill='blue', outline='blue')
            elif val == 0.05:
                self.w.create_oval((x*150)+35, (y*150)+35, ((x+1)
                                                            * 150)-35, ((y+1)*150)-35, fill='red', outline='red')
            elif val == 0.025:
                self.w.create_oval((x*150)+35, (y*150)+35, ((x+1)
                                                            * 150)-35, ((y+1)*150)-35, fill='orange', outline='orange')

        self.w.create_oval((pos[0]*150)+35, (pos[1]*150)+35, ((pos[0]+1)*150) -
                           35, ((pos[1]+1)*150)-35, fill='purple', outline='purple')

    def distance(self, start, end):
        max = abs(start[0]-end[0])
        if abs(start[1]-end[1]) > max:
            max = abs(start[1]-end[1])
        return max

    def observation_matrix(self):
        rows_observation = self.m*self.n*4
        columns_observation = rows_observation
        for m in range(self.n):
            for n in range(self.m):
                O_i = np.zeros((self.n, self.m))

                Observationmatrix = np.zeros(
                    (rows_observation, columns_observation))
                for k in range(self.m):
                    for l in range(self.n):
                        index = 4*k*self.n+4*l
                        if m == k and n == l:
                            O_i[k, l] = 0.1
                            for t in range(4):
                                Observationmatrix[index+t, index+t] = 0.1
                        elif self.distance((m, n), (k, l)) == 1:
                            O_i[k, l] = 0.05
                            for t in range(4):
                                Observationmatrix[index+t, index+t] = 0.05
                        elif self.distance((m, n), (k, l)) == 2:
                            O_i[k, l] = 0.025
                            for t in range(4):
                                Observationmatrix[index+t, index+t] = 0.025
                # print(O_i)

                self.O.append(Observationmatrix)
        O_i = np.zeros((self.m, self.n))
        for k in range(self.m):
            for l in range(self.n):
                index = 4*k*self.n+4*l
                if l == 0 or l == (self.n-1):
                    if k == 0 or k == (self.m-1):
                        O_i[k, l] = 0.625
                        for t in range(4):
                            Observationmatrix[index+t, index+t] = 0.625
                    else:
                        O_i[k, l] = 0.5
                        for t in range(4):
                            Observationmatrix[index+t, index+t] = 0.5
                elif k == 0 or k == (self.m-1):
                    O_i[k, l] = 0.5
                    for t in range(4):
                        Observationmatrix[index+t, index+t] = 0.5
                else:
                    O_i[k, l] = 0.325
                    for t in range(4):
                        Observationmatrix[index+t, index+t] = 0.325

        self.O.insert(0, Observationmatrix)

    def get_obs_matrix(self):
        '''
        Gives the observationmatrix to the corresponding position

        :param rows: number of rows
        :param columns: number of columns
        :param position: position of the robot (x,y)

        :return observation_matirx:
        '''
        position = self.robot.return_pos()

        self.observation_matrix()
        observation_matrix = self.choose_observationmatrix(position[0:2])
        observation_matrix = np.diag(observation_matrix)
        observation_matrix = observation_matrix[::4]
        observation_matrix = np.reshape(observation_matrix, (self.m, self.n))

        return observation_matrix

    def choose_observationmatrix(self, evidence):
        if evidence == (-1, -1):
            observationmatrix = self.O[0]
        else:
            observationmatrix = self.O[evidence[0]*self.n + evidence[1]+1]

        return observationmatrix

    def start_game(self):
        self.w.create_rectangle(0, 0, 600, 600, fill="white", outline='white')
        self.w.bind("<Button-1>", self.return_callback)
        for i in range(0, 4):
            for j in range(0, 4):
                self.w.create_rectangle(i*150, j*150, (i+1)*150,
                                        (j+1)*150, fill="white", outline='black')
        self.set_start_index()
        self.w.pack()
        self.master.mainloop()


new_game = OthelloGame()

new_game.start_game()
