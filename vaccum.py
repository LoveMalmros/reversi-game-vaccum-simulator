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
	board = np.zeros((4, 4))
	w = Canvas(master, width=600, height=600)
	robot = robot.Robot(4,4)
	robot.init_robot()        

	def set_start_index(self):
		self.index = randint(0, 15)

	def return_callback(self,event):
		print('hej')
		self.robot.move()
		self.render_grid()

	def render_grid(self):
		pos = self.robot.return_pos()
		print(self.robot.sensor())
		for i in range(0,4):
			for j in range(0,4):
				self.w.create_rectangle(i*150, j*150, (i+1)*150, (j+1)*150, fill="white", outline = 'black')
		self.w.create_oval((pos[0]*150)+35,(pos[1]*150)+35,((pos[0]+1)*150)-35,((pos[1]+1)*150)-35,fill='red', outline = 'red')


	def start_game(self):
		self.w.create_rectangle(0, 0, 600, 600, fill="white", outline = 'white')
		self.w.bind("<Button-1>", self.return_callback)
		for i in range(0,4):
			for j in range(0,4):
				self.w.create_rectangle(i*150, j*150, (i+1)*150, (j+1)*150, fill="white", outline = 'black')
		self.set_start_index()
		self.w.pack()
		self.master.mainloop()

new_game = OthelloGame()

new_game.start_game()

