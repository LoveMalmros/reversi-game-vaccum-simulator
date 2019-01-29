from tkinter import *
import numpy as np

BLACK_PIECE = 1
WHITE_PIECE = 2
EMPTY_PIECE = 0
POSSIBLE_PIECE = 3
AROUND_ARRAY = [1, 7, 8, 9]
TAKE_OPPONENT = 'take_opponent'
CALCULATE_POSSIBLE_MOVES = 'calculate_possible_moves'

class OthelloGame:
	master = Tk()
	w = Canvas(master, width=600, height=600)
	turn = WHITE_PIECE
	board = np.zeros((8, 8))
	possible_moves = set()
	score_var = StringVar()
	turn_var = StringVar()


	def changeHeader(self):
		white = 0
		black = 0
		for val in np.nditer(self.board):
			if val==BLACK_PIECE:
				black = black + 1
			if val==WHITE_PIECE:
				white = white + 1
		self.score_var.set('SCORE(white-black): ' + str(white) + ' - ' + str(black))
		if self.turn == WHITE_PIECE:
			self.turn_var.set('TURN: WHITE')
		else:
			self.turn_var.set('TURN: BLACK')

	def put_piece(self, x,y):
		if self.valid_move(x,y):
			newX = int(x / 75)
			newY = int(y / 75)
			index = newY * 8 + newX
			self.calculate_taken(index)
			color = self.get_color()
			np.put(self.board, index, self.turn)
			self.w.create_oval(newX*76,newY*76,(newX+1)*74,(newY+1)*74,fill=color, outline = color)
			self.turn = WHITE_PIECE if self.turn == BLACK_PIECE else BLACK_PIECE
			self.render_board_for_next_player()
			if(self.terminal_test()):
				print('Game finished! Neither player can make a valid move.')
				# TODO popup and show score
		else:
			print('Not a valid position. Please choose one of the red dots!')

	def put_piece_temp(self, x, y):
		if self.valid_move(x, y):
			newX = int(x / 75)
			newY = int(y / 75)
			index = newY * 8 + newX
			self.calculate_taken(index)
			'''
			color = self.get_color()
			np.put(self.board, index, self.turn)
			self.w.create_oval(newX*76, newY*76, (newX+1)*74,
			                   (newY+1)*74, fill=color, outline=color)
			'''
			self.turn = WHITE_PIECE if self.turn == BLACK_PIECE else BLACK_PIECE
			self.render_board_for_next_player()

	def valid_move(self, x,y):
		newX = int(x / 75)
		newY = int(y / 75)
		index = newY * 8 + newX
		return (index in self.possible_moves)

	def render_board_for_next_player(self):
		self.legalMoves()
		self.render_board()
		self.show_possible_moves()
		print(self.possible_moves)
		if(len(self.possible_moves) == 0):
			self.turn = WHITE_PIECE if self.turn == BLACK_PIECE else BLACK_PIECE
		self.changeHeader()


	def show_possible_moves(self):
		for idx in self.possible_moves:
			y = int(idx/8)
			x = idx - y*8
			self.w.create_oval((x*75)+35,(y*75)+35,((x+1)*75)-35,((y+1)*75)-35,fill='red', outline = 'red')

	def calculate_taken(self, index):
		for val in AROUND_ARRAY:
			self.adjacent_opponent(index, val, [], self.turn, TAKE_OPPONENT)
			self.adjacent_opponent(index, -val, [], self.turn, TAKE_OPPONENT)

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
					self.adjacent_opponent(i, val, [], self.turn, CALCULATE_POSSIBLE_MOVES)
					self.adjacent_opponent(i, -val, [], self.turn, CALCULATE_POSSIBLE_MOVES)

	def oppositeColor(self, color):
		if color == WHITE_PIECE:
			return BLACK_PIECE
		if color == BLACK_PIECE:
			return WHITE_PIECE
		return 0

	def check_if_inside(self, index, direction):
		y = int(index/8)
		x = index - y*8

		if(x==7 and (direction == 1 or direction == 9 or direction == -7)):
			return False
		if(x==0 and (direction == -1 or direction == 7 or direction == -9 )):
			return False
		if(y==7 and (direction == 7 or direction == 8 or direction == 9 )):
			return False
		if(y==0 and (direction == -7 or direction == -8 or direction == -9)):
			return False
		newIndex = index + direction
		return newIndex >= 0 and newIndex <8*8

	def adjacent_opponent(self, index, direction, current_adjacents, color, type):
		new_adjacents = current_adjacents
		if self.check_if_inside(index, direction) and np.take(self.board, index+direction) == self.oppositeColor(color):
			new_adjacents.append(index+direction)
			self.adjacent_opponent(index+direction, direction, new_adjacents, color, type)
		elif self.check_if_inside(index, direction) and len(new_adjacents)>0:
			if type == CALCULATE_POSSIBLE_MOVES and np.take(self.board, index+direction) == EMPTY_PIECE:
				if index+direction == 8:
					print(index)
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
			elif val == BLACK_PIECE:
				self.w.create_oval(col*76,row*76,(col+1)*74,(row+1)*74,fill="black", outline = 'black')
			else:
				self.w.create_rectangle(col*75, row*75, (col+1)*75, (row+1)*75, fill="green", outline = 'black')


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
		self.render_board_for_next_player()
		Label(self.master, textvariable=self.turn_var).pack()
		Label(self.master, textvariable=self.score_var).pack()
		self.w.pack()
		self.master.mainloop()

	def terminal_test(self): # when both players consecutively can't make any valid moves -> game finished!
		if(len(self.possible_moves) == 0):
			self.legalMoves() # check next player's possible moves
			if(len(self.possible_moves) == 0):
				return True
		else:
			return False


	def cutoff_test(self, depth):
		return (depth > 4 or self.terminal_test)
'''
	# FOR SOME INSIPIRATION <3
	
	def AlphaBeta(self, board, player, depth, alpha, beta, maximizingPlayer):
		if depth == 0 or self.terminal_test():
			return EvalBoard(board, player)
		if maximizingPlayer:
			v = minEvalBoard
			for y in range(n):
				for x in range(n):
					if self.valid_move(x, y):
						(boardTemp, totctr) = MakeMove(copy.deepcopy(board), x, y, player)
						v = max(v, AlphaBeta(boardTemp, player, depth - 1, alpha, beta, False))
						alpha = max(alpha, v)
						if beta <= alpha:
							break # beta cut-off
			return v
		else: # minimizingPlayer
			v = maxEvalBoard
			for y in range(n):
				for x in range(n):
					if self.valid_move(x, y):
						(boardTemp, totctr) = MakeMove(copy.deepcopy(board), x, y, player)
						v = min(v, AlphaBeta(boardTemp, player, depth - 1, alpha, beta, True))
						beta = min(beta, v)
						if beta <= alpha:
							break # alpha cut-off
			return v


	def search_pruning(state, game, d=4, cutoff_test = None, eval_f = None):
		def max_value(state, alpha, beta, depth):
			if cutoff_test(state, depth):
				return eval_f(state)
			v = float('inf')
			for idx in self.possible_moves:
				v = max(v, min_value())
'''


new_game = OthelloGame()

new_game.setupBoard()
