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

HEADINGS = [NORTH, EAST, SOUTH, WEST]

class HMM:
    def __init__(self,n,m ):
        self.master = Tk()
        self.m = m
        self.n = n
        self.matrix_dimension = m*n*4
        self.board_dimension = m*n
        self.board = np.zeros((n, m))
        self.w = Canvas(self.master, width=900, height=670)
        self.robot = robot.Robot(n, m)
        self.rect_dim = 600/n
        self.f_old = np.array([[1/(self.board_dimension)] for y in range(self.matrix_dimension)])
        self.probs = np.array([1/(self.matrix_dimension) for y in range(self.board_dimension)])
        self.O = []
        self.T = []
        self.nothing = True

    def set_start_index(self):
        self.index = randint(0, 15)

    def next_step(self):
        self.reset_board()
        self.forward_filtering()
        self.robot.move()
        self.render_grid()

    def reset_board(self):
        for i in range(0, self.n):
            for j in range(0, self.m):
                self.w.create_rectangle(i*self.rect_dim, j*self.rect_dim, (i+1)*self.rect_dim,
                                        (j+1)*self.rect_dim, fill="white", outline='black')

    def render_sensors(self):
        obs_matrix = self.get_obs_matrix()
        self.reset_board()
        x_pos = 0
        y_pos = 0
        for y, vec_p in enumerate(obs_matrix):
            for x, p in enumerate(vec_p):
                for pos in range(4):
                    if pos == 0:
                        x_pos = x*self.rect_dim + 15
                        y_pos = y*self.rect_dim + 55
                    elif pos == 1:
                        x_pos = x*self.rect_dim + 55
                        y_pos = y*self.rect_dim + 10
                    elif pos == 2:
                        x_pos = x*self.rect_dim + 100
                        y_pos = y*self.rect_dim + 55
                    elif pos == 3:
                        x_pos = x*self.rect_dim + 55
                        y_pos = y*self.rect_dim + 110
                    
                    self.w.create_text(x_pos,y_pos,fill="black",font="Times 10",
                        text=str(p))
                if(p==0.1):
                    self.w.create_rectangle(x_pos-10,y_pos-80, x_pos+15,y_pos-55,fill="blue")

    def render_trans(self):
        t_matrix = self.get_T_matrix()
        self.reset_board()
        x_pos = 0
        y_pos = 0
        for i,p in enumerate(t_matrix):
            pos = i % 4
            y = int(i/(4*self.m))
            x = int((i % (self.n*4))/4)
            if pos == 0:
                x_pos = x*self.rect_dim + 55
                y_pos = y*self.rect_dim + 15
            elif pos == 1:
                x_pos = x*self.rect_dim + 100
                y_pos = y*self.rect_dim + 55
            elif pos == 2:
                x_pos = x*self.rect_dim + 55
                y_pos = y*self.rect_dim + 100
            elif pos == 3:
                x_pos = x*self.rect_dim + 15
                y_pos = y*self.rect_dim + 55
                    
            self.w.create_text(x_pos,y_pos,fill="black",font="Times 10", text=str(p))

    def transition_matrix(self):
        T = np.zeros((self.matrix_dimension,self.matrix_dimension))
        for n in range(self.matrix_dimension):
            heading = n % 4
            index = int(n/4)
            self.calculate_transition_value_and_pos(index,heading,T,n)
        self.T = T

    def heading_to_grid_translation(self,heading):
        if heading == NORTH:
            return -self.m
        elif heading == EAST:
            return 1
        elif heading == SOUTH:
            return self.m
        elif heading == WEST:
            return -1
    
    def check_wall(self, curr_index, index):
        curr_y = int(curr_index/(4*self.m))
        new_y = int(index/(4*self.m))
        curr_x = int((curr_index % (4*self.m))/4)
        new_x = int((index % (4*self.m))/4)
        if (curr_y != new_y and curr_x != new_x) or index < 0 or index >= self.n*self.m*4:
            return True
        return False

    def calculate_transition_value_and_pos(self,index,heading,T,n): 
        other_headings = list(filter(lambda x: x != heading, HEADINGS))
        index_in_heading = (index + self.heading_to_grid_translation(heading))*4 + heading 
        encountering_wall = self.check_wall(n,index_in_heading)

        indices = []
        for h in other_headings:
            new_index = (index + self.heading_to_grid_translation(h))*4 + h 
            wall = self.check_wall(n,new_index)
            if not wall:
                indices.append(new_index)
        new_indices = list(filter(lambda x: x > -1 and x < self.n*self.m*4, indices))
        if encountering_wall and len(new_indices) > 0:
            prob = 1/len(new_indices)
            for i in new_indices:
                T[n, i] = prob
        elif len(new_indices) > 0:
            prob = 0.3/len(new_indices)
            for i in new_indices:
                T[n, i] = prob
            T[n,index_in_heading] = 0.7

    def go(self):
        total_distance = 0
        tot = 0
        for i in range(200):
            time.sleep( 0.03 )
            self.next_step()
            if(not self.nothing):
                estimate_index = self.probs.index(max(self.probs))
                x = int(estimate_index % self.n)
                y = int(estimate_index / self.n)
                distance = np.linalg.norm(np.array([self.robot.return_pos()[0], self.robot.return_pos()[1]]) - np.array([x,y]))
                total_distance = total_distance + distance
                tot = tot + 1
        print(total_distance/tot)
        
    def render_grid(self):
        pos = self.robot.return_pos()
        self.reset_board()
        probs = self.probs
        sensors = list(np.nditer(self.get_obs_matrix()))
        if 0.1 in sensors:
            sensor_index = sensors.index(0.1)
            self.nothing = False
        else:
            sensor_index = -1
        max_p = probs.index(max(probs))
        for i, val in enumerate(probs):
            x = int(i % self.n)
            y = int(i / self.n)
            if val > 0:
                self.w.create_rectangle(x*self.rect_dim, y*self.rect_dim, (x+1)*self.rect_dim, (y+1)*self.rect_dim, fill="yellow", outline='black')
            if val > 0.1:
                self.w.create_rectangle(x*self.rect_dim, y*self.rect_dim, (x+1)*self.rect_dim, (y+1)*self.rect_dim, fill="orange", outline='black')
            if i == max_p:
                self.w.create_rectangle(x*self.rect_dim, y*self.rect_dim, (x+1)*self.rect_dim, (y+1)*self.rect_dim, fill="grey", outline='black')
                self.w.create_rectangle(x*self.rect_dim+80,y*self.rect_dim+80, x*self.rect_dim+105,y*self.rect_dim+105,fill="red")
            if sensor_index != -1 and i == sensor_index:
                self.w.create_rectangle(x*self.rect_dim+40,y*self.rect_dim+40, x*self.rect_dim+65, y*self.rect_dim+65,fill="blue")
            if x == pos[0] and y == pos[1]:
                self.w.create_rectangle(pos[0]*self.rect_dim, pos[1]*self.rect_dim, (pos[0]+1)*self.rect_dim, (pos[1]+1)*self.rect_dim, fill="red", outline='black')
                self.w.create_rectangle(pos[0]*self.rect_dim+60,pos[1]*self.rect_dim+60, (pos[0])*self.rect_dim+85,(pos[1])*self.rect_dim+85,fill="black")
            self.w.create_text(x*self.rect_dim +70,y*self.rect_dim+8,fill="black",font="Times 10", text=str(val))
    
    def forward_filtering(self):
        probs = []
        i = 0
        prob = 0
        for p in self.f_old:
            prob = prob + p[0]
            if(i == 3):
                probs.append(prob)
                prob=0
                i=0
            else:
                i = i + 1
        self.probs = probs
        position = self.robot.return_pos()
        f_new = np.dot(np.dot(self.choose_observationmatrix(position[0:2]), np.transpose(self.T)), self.f_old)
        f_normalization = f_new / np.sum(f_new)
        self.f_old = f_normalization
        return f_normalization

    def distance(self, start, end):
        max = abs(start[0]-end[0])
        if abs(start[1]-end[1]) > max:
            max = abs(start[1]-end[1])
        return max


    def observation_matrix(self):
        for m in range(self.n):
            for n in range(self.m):
                O_i = np.zeros((self.n, self.m))

                Observationmatrix = np.zeros(
                    (self.matrix_dimension, self.matrix_dimension))
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
        pos = position[0]*self.n*4 + position[1]*4 + position[2]
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

    def start(self):
        self.observation_matrix()
        self.transition_matrix()
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
        b4 = Button(self.master, text="One step", command=self.next_step)
        b4.pack()
        b4.place(bordermode=OUTSIDE, x=180, y=625)
        b4.place(bordermode=OUTSIDE, height=30, width=70)
        b5 = Button(self.master, text="Go", command=self.go)
        b5.pack()
        b5.place(bordermode=OUTSIDE, x=260, y=625)
        b5.place(bordermode=OUTSIDE, height=30, width=70)
        self.master.mainloop()



hmm = HMM(5,5)

hmm.start()
