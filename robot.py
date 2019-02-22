import numpy as np
import collections

NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3

class Robot:


    # Constructor for Robot with playing field, field is all 0's
    # robot header begins at 0(North), with the rest being 1(East) 2(South) 3(West)
    def __init__(self, row, column):
        self.field = [[0 for x in range(0, row)] for y in range(0, column)]
        self.states = row*column
        self.row = row
        self.column = column
        self.header = 0


    # robot's position is chosen at random and will be 1 on the field
    def init_robot(self):
        randrow = np.random.choice(np.arange(0, self.row))
        randcol = np.random.choice(np.arange(0, self.column))

        self.field[randrow][randcol] = 1

        ## CALL CHECK HEADER
        self.check_header()

    # function to change header
    # for walls, corners, or neither
    def check_header(self):
        # gets position of robot
        posx = self.return_pos()[0]
        posy = self.return_pos()[1]

        # check to see if there has been a change made
        headerCheck = 0
        # Corners

        # Corner 0,0

        if posx == 0 and posy == 0:
            self.header = np.random.choice(np.arange(0, 4), p=[0, 1/2, 1/2, 0])
            headerCheck = 1

        # Corner 0, max(column)
        elif posx == 0 and posy == self.column-1:
            self.header = np.random.choice(np.arange(0,4), p=[0, 0, 1/2, 1/2])
            headerCheck = 1

        # Corner max(row), 0
        elif posx == self.row-1 and posy == 0:
            self.header = np.random.choice(np.arange(0, 4), p=[1/2, 1/2, 0, 0])
            headerCheck = 1

        # Corner max(row), max(col)
        elif posx == self.row-1 and posy == self.column-1:
            self.header = np.random.choice(np.arange(0, 4), p =[1/2, 0, 0, 1/2])
            headerCheck = 1


        # Walls
        if headerCheck == 0:
            # Facing Left with wall on the left
            if posy == 0:
                self.header = np.random.choice(np.arange(0,4), p =[1/3, 1/3, 1/3, 0])
                headerCheck = 2

            # Facing Right with wall on right
            elif posy == self.column-1:
                self.header = np.random.choice(np.arange(0, 4), p=[1/3, 0, 1/3, 1/3])
                headerCheck = 2

            # Facing Up with wall up
            elif posx == 0:
                self.header = np.random.choice(np.arange(0, 4), p =[0, 1/3, 1/3, 1/3])
                headerCheck = 2

            # Facing down with wall down
            elif posx == self.column-1:
                self.header = np.random.choice(np.arange(0, 4), p =[1/3, 1/3, 0, 1/3])
                headerCheck = 2

        # change of header if no walls
        # same direction == 70%, else 30%
        if headerCheck == 0:
            if self.header == 0:
                self.header = np.random.choice(np.arange(0, 4), p=[.7, .1, .1, .1])
            elif self.header == 1:
                self.header = np.random.choice(np.arange(0, 4), p=[.1, .7, .1, .1])
            elif self.header == 2:
                self.header = np.random.choice(np.arange(0, 4), p=[.1, .1, .7, .1])
            elif self.header == 3:
                self.header = np.random.choice(np.arange(0, 4), p=[.1, .1, .1, .7])



    # returns actual position
    def return_pos(self):

        for x in range(0, self.row):
            for y in range(0, self.column):
                if self.field[x][y] == 1:
                    return x, y, self.header


    # movement of the robot
    def move(self):
        # gets position of robot
        posx = self.return_pos()[0]
        posy = self.return_pos()[1]
        print("X: " + str(posx) + " Y: "+ str(posy) )
        # checks the header so it can head in the correct
        self.check_header()

        # makes a forward move based on position
        # original state is turned to 0
        # new state is 1
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
        

    def ret_field(self):
        return self.field

    # returns the manhattan distance for a certain point to the location of the robot
    def manhattan_distance(self, x, y):
        posx = self.return_pos()[0]
        posy = self.return_pos()[1]

        # Takes absolute value of differences
        diffx = abs(x-posx)
        diffy = abs(y-posy)

        # Calculates Manhattan distance
        totaldiff = diffx + diffy

        return totaldiff

    # create a list with numbers 0=> (0,0), 1=> (0,1), ...etc
    def create_dict(self):
        pos = self.return_pos()

        dict = collections.OrderedDict()
        count1 = 0
        count2 = 0
        arrangedlist = []

        for x in range(0, self.row):
            for y in range(0, self.column):
                xy = [x, y]
                if self.ret_distance(pos, xy) == 0:
                    dict.update({(x,y): 0})
                elif self.ret_distance(pos, xy) == 1:
                    dict.update({(x, y): 1})
                    count1 += 1
                elif self.ret_distance(pos, xy) == 2:
                    dict.update({(x, y): 2})
                    count2 += 1
                else:
                    dict.update({(x, y): 3})

        newdict = collections.OrderedDict(sorted(dict.items(), key=lambda t: t[1]))

        # converts dictionary into a list
        while len(newdict) != 0:
            tempxy = newdict.popitem(False)[0]
            arrangedlist.append(tempxy)

        return arrangedlist, count1, count2



    # returns distance between 2 points
    def ret_distance(self, start, end):
        max = abs(start[0] - end[0])
        if abs(start[1] - end[1]) > max:
            max = abs(start[1]-end[1])
        return max



    # returns the position with highest probability [0]

    # returns vector with positions LS [1]
    # returns vector with positions LS2 [2]
    def sensor(self):
        # Dictionary with sensor values arranged by their weights
        # converted to a list; [0] is the location, [1] is Ls value, [2] is Ls value etc...
        dict = self.create_dict()[0]
        count1 = self.create_dict()[1]
        count2 = self.create_dict()[2]

        # number of states in the location, ls1, and ls2 + nothing
        numprobability = 1+count1+count2+1

        # new list with number of states = location + Ls + Ls2 + 1 for nothing state
        statelist = list(range(0, numprobability))

        # Nothing probability
        nothingprob = 1 - .1 - count1 * .05 - count2 * .025

        # probability list
        prob = [.1]
        for x in range(0, count1):
            prob.append(.05)
        for x in range(0, count2):
            prob.append(.025)
        # append nothing probability
        prob.append(nothingprob)

        # chooses the index of location from the list of states
        proboflocation = np.random.choice(statelist, p=prob)

        # checks to see if nothing probability is chosen
        if proboflocation == numprobability - 1:
            sensorlocation = (-1, -1)
        else:
            # DUMMYYYY needs to be changed
            sensorlocation = dict[proboflocation]
            #print(sensorlocation)

        return sensorlocation




