from tkinter import *
import numpy as np

BLACK_PIECE = 1
WHITE_PIECE = 2
EMPTY_PIECE = 0
AROUND_ARRAY = [1, 7, 8, 9]
TAKE_OPPONENT = 'take_opponent'
CALCULATE_POSSIBLE_MOVES = 'calculate_possible_moves'

class OthelloGame:
	master = Tk()
	w = Canvas(master, width=600, height=600)
	turn = WHITE_PIECE
	board = np.zeros((8, 8))
	possible_moves = set()


	def put_piece(self, x,y):
		newX = int(x / 75)
		newY = int(y / 75)
		index = newY * 8 + newX
		if self.checkIfTaken(index) == False and index in self.possible_moves:
			self.calculate_taken(index)

			color = self.get_color()
			np.put(self.board, index, self.turn)
			self.w.create_oval(newX*76,newY*76,(newX+1)*74,(newY+1)*74,fill=color, outline = color)
			self.turn = WHITE_PIECE if self.turn == BLACK_PIECE else BLACK_PIECE

			self.legalMoves()
			self.render_board()

	def calculate_taken(self, index):
		for val in AROUND_ARRAY:
			self.adjacentOpponent(index, val, [], self.turn, TAKE_OPPONENT)
			self.adjacentOpponent(index, -val, [], self.turn, TAKE_OPPONENT)

	def get_color(self):
		return 'white' if self.turn == WHITE_PIECE else 'black'

	def callback(self,event):
		self.put_piece(event.x, event.y)

	def checkIfTaken(self, i):
		return np.take(self.board,i) == WHITE_PIECE or np.take(self.board,i) == BLACK_PIECE

	def legalMoves(self):
		self.possible_moves = set()
		#Filter self.board for turns pieces
		for i,val in enumerate(np.nditer(self.board)):
			if val == self.turn:
				for val in AROUND_ARRAY:
					self.adjacentOpponent(i, val, [], self.turn, CALCULATE_POSSIBLE_MOVES)
					self.adjacentOpponent(i, -val, [], self.turn, CALCULATE_POSSIBLE_MOVES)

	def oppositeColor(self, color):
		if color == WHITE_PIECE:
			return BLACK_PIECE
		if color == BLACK_PIECE:
			return WHITE_PIECE
		return 0

	def adjacentOpponent(self,index,direction, current_adjacents, color, type):
		new_adjacents = current_adjacents
		if index+direction >= 0 and index+direction < 8*8 and np.take(self.board, index+direction) == self.oppositeColor(color):
			new_adjacents.append(index+direction)
			self.adjacentOpponent(index+direction, direction, new_adjacents, color, type)
		elif index+direction >= 0 and index+direction < 8*8 and len(new_adjacents)>0:
			if type == CALCULATE_POSSIBLE_MOVES and np.take(self.board, index+direction) == EMPTY_PIECE:
				self.possible_moves.add(index+direction)
			elif type == TAKE_OPPONENT and np.take(self.board, index+direction) == color:
				for idx in current_adjacents:
					np.put(self.board, idx, color)

	def render_board(self):
		for i,val in enumerate(np.nditer(self.board)):
			row = int(i/8)
			col = i - row*8
			if val == WHITE_PIECE:
				self.w.create_oval(col*76,row*76,(col+1)*74,(row+1)*74,fill="white", outline = 'white')
			if val == BLACK_PIECE:
				self.w.create_oval(col*76,row*76,(col+1)*74,(row+1)*74,fill="black", outline = 'black')

	def setupBoard(self):

		self.w.create_rectangle(0, 0, 600, 600, fill="green", outline = 'green')
		self.w.bind("<Button-1>", self.callback)
		for i in range(0,8):
			for j in range(0,8):
				self.w.create_rectangle(i*75, j*75, (i+1)*75, (j+1)*75, fill="green", outline = 'black')
		np.put(self.board, 27, WHITE_PIECE)
		np.put(self.board, 28, BLACK_PIECE)
		np.put(self.board, 35, BLACK_PIECE)
		np.put(self.board, 36, WHITE_PIECE)
		self.render_board()
		self.legalMoves()
		print(self.possible_moves)
		self.w.pack()
		self.master.mainloop()

new_game = OthelloGame()

new_game.setupBoard()
