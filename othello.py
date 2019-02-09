from tkinter import *
import numpy as np
import operator
import copy


BLACK_PIECE = 1
WHITE_PIECE = 2
EMPTY_PIECE = 0
AROUND_ARRAY = [1, 7, 8, 9]
TAKE_OPPONENT = 'take_opponent'
CALCULATE_POSSIBLE_MOVES = 'calculate_possible_moves'
PLAYER_VS_PLAYER = 'player_vs_player'
PLAYER_VS_AI = 'player_vs_ai'


class OthelloGame:
	master = Tk()
	w = Canvas(master, width=600, height=600)
	turn = WHITE_PIECE
	board = np.zeros((8, 8))
	possible_moves = {}
	score_var = StringVar()
	turn_var = StringVar()
	TYPE_OF_GAME = ''
	PLAYING_AS = 0
	DEPTH = 1

	def menu(self):
		self.type_of_game();
		if(self.TYPE_OF_GAME == PLAYER_VS_AI):
			self.player_color()
			self.depth_of_max_min()


	def depth_of_max_min(self):
		print('What depth should the ais tree search alogrithm have?(1-7)')
		depth = input()
		if(int(depth)>0 and int(depth) < 8):
			self.DEPTH = int(depth)
		else:
			print('Not a valid depth')
			self.depth_of_max_min()

	def player_color(self):
		print('What color would you like to play as?')
		print('1. White')
		print('2. Black')
		color = input()
		if(color == '1'):
			self.PLAYING_AS = WHITE_PIECE
		elif(color == '2'):
			self.PLAYING_AS = BLACK_PIECE
		else:
			print('Not a valid choice')
			self.player_color()

	def type_of_game(self):
		print('Play against ai or player?')
		print('1. Player')
		print('2. Ai')
		type_of_game = input()
		if(type_of_game == '1'):
			self.TYPE_OF_GAME = PLAYER_VS_PLAYER
		elif(type_of_game == '2'):
			self.TYPE_OF_GAME = PLAYER_VS_AI
		else:
			print('Not a valid choice')
			self.type_of_game()

	def my_turn_vs_ai(self):
		if(self.TYPE_OF_GAME == PLAYER_VS_AI and self.turn == self.PLAYING_AS):
			return True
		return False

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


	def put_piece(self, index):
		if self.valid_move(index, self.possible_moves):
			self.calculate_taken(index, self.board, self.turn)
			np.put(self.board, index, self.turn)
			color = self.get_color()
			self.turn = WHITE_PIECE if self.turn == BLACK_PIECE else BLACK_PIECE
			self.possible_moves = {}
			self.legalMoves(self.possible_moves, self.board, self.turn) #GET POSSBLE MOVES
			self.render_board()

		else:
			print('Not a valid position. Please choose one of the red dots!')

	def callback(self,event):
		if(self.my_turn_vs_ai()):
			index, x, y = self.get_index(event.x, event.y)
			self.put_piece(index)
		else:
			print('AIs turn!')

	def eval_board(self, board, color):
		points = 0
		print(board)
		for i,val in enumerate(np.nditer(board)):
			if val == color:
				points = points + 1
		print(points)
		return points

	def alpha_beta(self, board, depth, alpha, beta, possible_moves, max_player, color, path_and_value):
		self.legalMoves(possible_moves, board, color)
		if depth > self.DEPTH:
			return self.eval_board(board, self.oppositeColor(self.PLAYING_AS))
		if max_player:
			v = -1000
			for move_index, value in possible_moves.items():
				np.put(board, move_index, color)
				self.calculate_taken(int(move_index), board, color)
				v = max(v, self.alpha_beta(copy.deepcopy(board), depth + 1, alpha, beta, {}, False, self.oppositeColor(color), path_and_value))
				alpha = max(alpha, v)
				if move_index in path_and_value and path_and_value[move_index] < alpha:
					path_and_value[move_index] = alpha
				else:
					path_and_value[move_index] = alpha
				if beta <= alpha:
					break
			return v
		else:
			v = 1000
			for move_index, value in possible_moves.items():
				np.put(board, move_index, color)
				self.calculate_taken(int(move_index), board, color)
				v = min(v, self.alpha_beta(copy.deepcopy(board), depth + 1, alpha, beta, {}, True, self.oppositeColor(color), path_and_value))
				beta = min(beta, v)
				#if  path_and_value[move_index] and path_and_value[move_index] > beta:
				#	path_and_value[move_index] = beta
				if beta <= alpha:
					break
			return v

	def get_index(self, x, y):
		new_x = int(x / 75)
		new_y = int(y / 75)
		index = new_y * 8 + new_x
		return index, new_x, new_y

	def valid_move(self, index, possible_moves):
		return (str(index) in possible_moves)

	def show_possible_moves(self):
		for idx in self.possible_moves:
			y = int(int(idx)/8)
			x = int(idx) - y*8
			self.w.create_oval((x*75)+35,(y*75)+35,((x+1)*75)-35,((y+1)*75)-35,fill='red', outline = 'red')

	def calculate_taken(self, index, board, turn):
		for val in AROUND_ARRAY:
			self.adjacent_opponent(index, val, [], turn, TAKE_OPPONENT, {}, board)
			self.adjacent_opponent(index, -val, [], turn, TAKE_OPPONENT, {}, board)

	def get_color(self):
		return 'white' if self.turn == WHITE_PIECE else 'black'



	def checkIfTaken(self, i):
		return np.take(self.board,i) == WHITE_PIECE or np.take(self.board,i) == BLACK_PIECE

	def legalMoves(self, possible_moves, board, turn):
		#Filter self.board for turns pieces
		for i,val in enumerate(np.nditer(board)):
			if val == turn:
				for dir in AROUND_ARRAY:
					self.adjacent_opponent(i, dir, [], turn, CALCULATE_POSSIBLE_MOVES, possible_moves, board)
					self.adjacent_opponent(i, -dir, [], turn, CALCULATE_POSSIBLE_MOVES, possible_moves, board)

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

	def adjacent_opponent(self, index, direction, current_adjacents, color, type, possible_moves, board):
		new_adjacents = current_adjacents
		new_index = index + direction
		if self.check_if_inside(index, direction) and np.take(board, index+direction) == self.oppositeColor(color):
			new_adjacents.append(new_index)
			self.adjacent_opponent(new_index, direction, new_adjacents, color, type, possible_moves, board)
		elif self.check_if_inside(index, direction) and len(new_adjacents)>0:
			if type == CALCULATE_POSSIBLE_MOVES and np.take(board, new_index) == EMPTY_PIECE:
				self.add_index_to_moves(new_index, possible_moves, len(current_adjacents))
			elif type == TAKE_OPPONENT and np.take(board, new_index) == color:
				for idx in current_adjacents:
					np.put(board, idx, color)

	def add_index_to_moves(self, index, pos_moves, number_of_moves):
		if str(index) in pos_moves:
			pos_moves[str(index)] = int(pos_moves[str(index)]) + number_of_moves
		else:
			pos_moves[str(index)] = number_of_moves

	def render_board(self):
		self.changeHeader()
		if(len(self.possible_moves) == 0):
			self.turn = WHITE_PIECE if self.turn == BLACK_PIECE else BLACK_PIECE
			print('There was no valid move. Switched to other player\'s turn.')
			self.changeHeader()
			if(self.terminal_test(self.possible_moves, self.board, self.turn)):
				print('Game finished! Neither player can make a valid move.')
				print(self.score_var.get())
		for i,val in enumerate(np.nditer(self.board)):
			row = int(i/8)
			col = i - row*8
			if val == WHITE_PIECE:
				self.w.create_oval(col*76,row*76,(col+1)*74,(row+1)*74,fill="white", outline = 'white')
			elif val == BLACK_PIECE:
				self.w.create_oval(col*76,row*76,(col+1)*74,(row+1)*74,fill="black", outline = 'black')
			else:
				self.w.create_rectangle(col*75, row*75, (col+1)*75, (row+1)*75, fill="green", outline = 'black')
		self.show_possible_moves()
		if(self.TYPE_OF_GAME == PLAYER_VS_AI and self.turn != self.PLAYING_AS):
			path_and_value = {}
			self.alpha_beta(copy.deepcopy(self.board), 0, -1000, 1000, {}, True, self.turn, path_and_value)
			print(path_and_value)
			idx = int(max(path_and_value.items(), key=operator.itemgetter(1))[0])
			self.put_piece(idx)


	def start_game(self):
		self.menu()
		self.w.create_rectangle(0, 0, 600, 600, fill="green", outline = 'green')
		self.w.bind("<Button-1>", self.callback)
		for i in range(0,8):
			for j in range(0,8):
				self.w.create_rectangle(i*75, j*75, (i+1)*75, (j+1)*75, fill="green", outline = 'black')
		np.put(self.board, 27, WHITE_PIECE)
		np.put(self.board, 28, BLACK_PIECE)
		np.put(self.board, 35, BLACK_PIECE)
		np.put(self.board, 36, WHITE_PIECE)
		self.legalMoves(self.possible_moves, self.board, self.turn) #GET POSSBLE MOVES
		self.render_board()
		Label(self.master, textvariable=self.turn_var).pack()
		Label(self.master, textvariable=self.score_var).pack()
		self.w.pack()
		self.master.mainloop()

	def terminal_test(self, possible_moves, board, turn): # when both players consecutively can't make any valid moves -> game finished!
		if(len(possible_moves) == 0):
			self.legalMoves(possible_moves, board, turn) # check next player's possible moves
			if(len(possible_moves) == 0):
				return True
		else:
			return False

new_game = OthelloGame()

new_game.start_game()
