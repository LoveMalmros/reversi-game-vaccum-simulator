
# Imports

import numpy as np

# HMM 

class hmm():

    def __init__(self, m,n):


        self.m = m # amount of rows
        self.n = n # amount of columns
        self.T = self.transition_matrix()
        self.O = self.observation_matrix()        
    

    def transition_matrix(self):
        '''
        calculates the transistion matrix
        
        '''
        rows_transition = self.m*self.m*4
        columns_transition = rows_transition
        T = np.zeros((rows_transition,columns_transition))

        for n in range(rows_transition):
            
            t = n % 4 # gives the heading t = 0: north t = 1 East t = 2 South T=3 West
            direction_change = False
            headings = [False,False,False,False]
            walls = 0

            if n-4*self.m< 0:
                headings[0] =  0
                walls += 1
                if t == 0:
                    direction_change = True
            #east
            if n % ( 4 * self.n ) + 4  >= 4 * self.m :
                headings[1] = 0
                walls += 1
                if t == 1:
                    direction_change = True
            #south
            if n // (4*self.n ) + 1 >= self.m:
                headings[2] = 0
                walls += 1
                if t == 2:
                    direction_change = True
            #west

            if n % ( 4 * self.n ) - 4 < 0:
                headings[3] = 0
                walls += 1
                if t == 3:
                    direction_change = True
    
            for i,e in enumerate(headings):
         
                if direction_change:
                    if e is False:
                        headings[i] = 1 / ( 4 - walls )
                elif e is False: 
                
                    if i == t:
                        headings[i] = 0.7
                    else:
                        headings[i] = 0.3 / ( 3 - walls )
    
            #north           
            try:
                if headings[0]!= 0 :
                    T[n ,( n - t ) - 4 * self.n ] =  headings[0] 
            except IndexError:
                pass

            #east
        
            try:
                if headings[1] != 0:
                    T[n,(n-t+1) +4] = headings[1]
            except IndexError:
                pass
        
            #south
            try:
                if headings[2] != 0:
                    T[n,(n-t+2)+4*self.n] = headings[2] 
            except IndexError:
                pass
            #west
    
            try:
                if headings[3] != 0:
                    T[n,n-t+3-4] = headings[3]
            except IndexError:
                pass

        return T



    def distance(self, start,end):
        max = abs(start[0]-end[0])
        if abs(start[1]-end[1]) > max:
            max = abs(start[1]-end[1])
        return max

    def choose_observationmatrix(self, evidence ):           
        if evidence == (-1,-1):
            observationmatrix = self.O[0]
        else: 
             observationmatrix = self.O[evidence[0]*self.n + evidence[1]+1] 
        print(observationmatrix)
        return observationmatrix


    def observation_matrix(self):
        rows_observation = self.m*self.n*4
        columns_observation = rows_observation
        O = []
        for m in range(self.n):
            for n in range(self.m):
                O_i = np.zeros((self.n,self.m))
                
                Observationmatrix = np.zeros((rows_observation,columns_observation))
                for k in range(self.m):
                    for l in range(self.n):
                        index = 4*k*self.n+4*l
                        if m==k and n==l:
                            O_i[k,l] = 0.1
                            for t in range(4):
                                Observationmatrix[index+t,index+t] = 0.1
                        elif self.distance((m,n),(k,l)) == 1:
                            O_i[k,l] = 0.05
                            for t in range(4):
                                Observationmatrix[index+t,index+t] = 0.05
                        elif self.distance((m,n),(k,l)) == 2:
                            O_i[k,l] = 0.025
                            for t in range(4):
                                Observationmatrix[index+t,index+t] = 0.025
                #print(O_i)   
            
                O.append(Observationmatrix)
        O_i = np.zeros((self.m,self.n))
        for k in range(self.m):
            for l in range(self.n):
                index = 4*k*self.n+4*l
                if l == 0 or l== (self.n-1):
                    if k == 0 or k == (self.m-1):
                        O_i[k,l] = 0.625
                        for t in range(4):
                            Observationmatrix[index+t,index+t] = 0.625
                    else:
                        O_i[k,l] = 0.5
                        for t in range(4):
                            Observationmatrix[index+t,index+t] = 0.5
                elif k == 0 or k == (self.m-1):
                    O_i[k,l] = 0.5
                    for t in range(4):
                        Observationmatrix[index+t,index+t] = 0.5
                else:
                    O_i[k,l] = 0.325
                    for t in range(4):
                        Observationmatrix[index+t,index+t] = 0.325

        O.insert(0, Observationmatrix)
                    
                        
       
        return O
            
        

    def forward_filtering(self,f_old ,evidence):
        '''
        description

        :param e: the current evidence for time step t (position)
        '''

        
        f_new = np.dot(np.dot(self.choose_observationmatrix(evidence ), (self.T).T),f_old)
        f_normalization = f_new / np.sum(f_new)

        


        return f_normalization

#m = 3
#n = 3

#dimension = m * n * 4
#probability = 1/(dimension)
#f_old = np.array([[probability] for y in range(dimension)])
#print(f_old)
#hmm_test = hmm(m,n)
#print(hmm_test.forward_filtering(f_old,(0,0)))
