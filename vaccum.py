from tkinter import *
import numpy as np
import operator
import copy
import time
import robot
from random import randint
from hmm import hmm



NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3

HEADINGS = [NORTH, EAST, SOUTH, WEST]

class OthelloGame:
    master = Tk()
    m = 4
    n = 4
    board = np.zeros((m, n))
    w = Canvas(master, width=900, height=670)
    robot = robot.Robot(4, 4)
    robot.init_robot()
    hiddemmm = hmm(4,4)
    #start_probability = 1/(4*4*4)
    #f_old = np.array([[start_probability] for y in range(4*4*4)])
    #hiddemmm.forward_filtering(f_old, robot.sensor())
    O = []
    T = []

    def set_start_index(self):
        self.index = randint(0, 15)

    def next_step(self):
        self.reset_board()
        self.robot.move()
        self.render_grid()

    def reset_board(self):
        for i in range(0, 4):
            for j in range(0, 4):
                self.w.create_rectangle(i*150, j*150, (i+1)*150,
                                        (j+1)*150, fill="white", outline='black')

    def render_sensors(self):
        obs_matrix = self.get_obs_matrix()
        self.reset_board()
        x_pos = 0
        y_pos = 0
        for y, vec_p in enumerate(obs_matrix):
            for x, p in enumerate(vec_p):
                for pos in range(4):
                    if pos == 0:
                        x_pos = x*150 + 15
                        y_pos = y*150 + 70
                    elif pos == 1:
                        x_pos = x*150 + 70
                        y_pos = y*150 + 10
                    elif pos == 2:
                        x_pos = x*150 + 130
                        y_pos = y*150 + 70
                    elif pos == 3:
                        x_pos = x*150 + 70
                        y_pos = y*150 + 140
                    
                    self.w.create_text(x_pos,y_pos,fill="black",font="Times 10",
                        text=str(p))

    def render_trans(self):
        obs_matrix = self.get_T_matrix()
        self.reset_board()
        x_pos = 0
        y_pos = 0
        for i,p in enumerate(obs_matrix):
            pos = i % 4
            y = int(i/16)
            x = int((i % 16)/4)
            if pos == 0:
                x_pos = x*150 + 15
                y_pos = y*150 + 70
            elif pos == 1:
                x_pos = x*150 + 70
                y_pos = y*150 + 10
            elif pos == 2:
                x_pos = x*150 + 130
                y_pos = y*150 + 70
            elif pos == 3:
                x_pos = x*150 + 70
                y_pos = y*150 + 140
                    
            self.w.create_text(x_pos,y_pos,fill="black",font="Times 10", text=str(p))

    def transition_matrix(self):
        rows_transition = 4*4*4
        columns_transition = 4*4*4
        T = np.zeros((rows_transition,columns_transition))
        for n in range(rows_transition):
            heading = n % 4
            index = int(n/4)
            self.calculate_transition_value_and_pos(index,heading,T,n)
        self.T = T

    def heading_to_grid_translation(self,heading):
        if heading == NORTH:
            return -4
        elif heading == EAST:
            return 1
        elif heading == SOUTH:
            return 4
        elif heading == WEST:
            return -2
    
    def check_wall(self, curr_index, index):
        curr_y = int(curr_index/16)
        new_y = int(index/16)
        curr_x = int((curr_index % 16)/4)
        new_x = int((index % 16)/4)
        if (curr_y != new_y and curr_x != new_x) or index < 0 or index > 63:
            return True
        return False

    def calculate_transition_value_and_pos(self,index,heading,T,n): 
        other_headings = list(filter(lambda x: x != heading, HEADINGS))
        index_in_heading = (index + self.heading_to_grid_translation(heading))*4 + heading + 1
        encountering_wall = self.check_wall(n,index_in_heading)

        indices = []
        for h in other_headings:
            new_index = (index + self.heading_to_grid_translation(h))*4 + h + 1
            wall = self.check_wall(n,new_index)
            if not wall:
                indices.append(new_index)
        new_indices = list(filter(lambda x: x > -1 and x < 64, indices))
        if encountering_wall and len(new_indices) > 0:
            prob = 1/len(new_indices)
            for i in new_indices:
                T[n, i] = prob
        elif len(new_indices) > 0:
            prob = 0.3/len(new_indices)
            for i in new_indices:
                T[n, i] = prob
            T[n,index_in_heading] = 0.7

        
    def render_grid(self):
        pos = self.robot.return_pos()
        # print(self.robot.sensor())
        self.reset_board()
        for i, val in enumerate(np.nditer(self.get_obs_matrix())):
            x = int(i % self.n)
            y = int(i / self.n)
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
        observation_matrix = self.choose_observationmatrix(position[0:2])
        observation_matrix = np.diag(observation_matrix)
        observation_matrix = observation_matrix[::4]
        observation_matrix = np.reshape(observation_matrix, (self.m, self.n))

        return observation_matrix

    def get_T_matrix(self):
        position = self.robot.return_pos()
        pos = position[0]*16 + position[1]*4 + position[2]
        self.transition_matrix()
        T_matrix = self.choose_t_matrix(pos)
        return T_matrix

    def choose_t_matrix(self, pos):
        return self.T[pos]

    def choose_observationmatrix(self, evidence):
        if evidence == (-1, -1):
            observationmatrix = self.O[0]
        else:
            observationmatrix = self.O[evidence[0]*self.n + evidence[1]+1]

        return observationmatrix

    def start_game(self):
        self.observation_matrix()
        self.w.create_rectangle(0, 0, 600, 600, fill="white", outline='white')
        self.reset_board()
        self.set_start_index()
        self.w.pack()
        b = Button(self.master, text="Show Transitions", command=self.render_trans)
        b.pack()
        b.place(bordermode=OUTSIDE, x=600, y=0)
        b.place(bordermode=OUTSIDE, height=300, width=300)
        b2 = Button(self.master, text="Show Sensors", command=self.render_sensors)
        b2.pack()
        b2.place(bordermode=OUTSIDE, x=600, y=300)
        b2.place(bordermode=OUTSIDE, height=300, width=300)
        b3 = Button(self.master, text="Init filter", command=self.next_step)
        b3.pack()
        b3.place(bordermode=OUTSIDE, x=100, y=625)
        b3.place(bordermode=OUTSIDE, height=30, width=70)
        b4 = Button(self.master, text="One step", command=self.next_step)
        b4.pack()
        b4.place(bordermode=OUTSIDE, x=180, y=625)
        b4.place(bordermode=OUTSIDE, height=30, width=70)
        b5 = Button(self.master, text="Go", command=self.next_step)
        b5.pack()
        b5.place(bordermode=OUTSIDE, x=260, y=625)
        b5.place(bordermode=OUTSIDE, height=30, width=70)
        b6 = Button(self.master, text="Stop", command=self.next_step)
        b6.pack()
        b6.place(bordermode=OUTSIDE, x=340, y=625)
        b6.place(bordermode=OUTSIDE, height=30, width=70)
        self.master.mainloop()



new_game = OthelloGame()

new_game.start_game()
