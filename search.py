# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from custom_types import Direction
from pacman import GameState
from typing import Any, Tuple,List
import util

"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self)->Any:
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state:Any)->bool:
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state:Any)->List[Tuple[Any,Direction,int]]:
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions:List[Direction])->int:
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()



def tinyMazeSearch(problem:SearchProblem)->List[Direction]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem:SearchProblem)->List[Direction]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    start = problem.getStartState() # We get the initial state

    To_visit = util.Stack()       # Our fringe keeping the state to visit under the form:
    To_visit.push([start,[]])     # [state, path]

    Visited = [] # The list of the visited state (graph search)

    while not To_visit.isEmpty() :
        position = To_visit.pop()
        # Even if we check before adding the state to the fringe if it has been visited
        # We check again whe we take it form it in case it has been visited since
        while position[0] in Visited:
            position = To_visit.pop()

        if problem.isGoalState(position[0]): # If the state is the solution we return the path
            return(position[1])
        else : # In the other case we expand the node and add non visited neighbors to the fringe
            Successors = problem.getSuccessors(position[0])
            for neighbor in Successors :
                if not(neighbor[0] in Visited):
                    To_visit.push([neighbor[0],position[1] + [neighbor[1]]])

            Visited.append(position[0]) # We add the expanded state to the list of the visited ones

    # If the search do not find any solution the algorithm return None
    return None

def breadthFirstSearch(problem:SearchProblem)->List[Direction]:
    """Search the shallowest nodes in the search tree first."""

    ###################################################################################
    # We use the same code as before but replacing the Stack (LIFO) by a Queue (FIFO) #
    ###################################################################################

    start = problem.getStartState() # We get the initial state

    To_visit = util.Queue()       # Our fringe keeping the state to visit under the form:
    To_visit.push([start,[]])     # [state, path]

    Visited = [] # The list of the visited state (graph search)

    while not To_visit.isEmpty() :
        position = To_visit.pop()
        # Even if we check before adding the state to the fringe if it has been visited
        # We check again whe we take it form it in case it has been visited since
        while position[0] in Visited:
            position = To_visit.pop() 

        if problem.isGoalState(position[0]) : # If the state is the solution we return the path
            return(position[1])
        else : # In the other case we expand the node and add non visited neighbors to the fringe
            Successors = problem.getSuccessors(position[0])
            for neighbor in Successors :
                if not(neighbor[0] in Visited):
                    To_visit.push([neighbor[0],position[1] + [neighbor[1]]])

            Visited.append(position[0]) # We add the expanded state to the list of the visited ones

    # If the search do not find any solution the algorithm return None
    return None

def uniformCostSearch(problem:SearchProblem)->List[Direction]:
    """Search the node of least total cost first."""

    #####################################################################################
    # We use the same code as before but replacing the Queue (FIFO) by a Priority Queue #
    #                   By computing the cost of each path                              #
    #####################################################################################

    start = problem.getStartState() # We get the initial state

    To_visit = util.PriorityQueue()       # Our fringe keeping the state to visit under the form:
    To_visit.push([start,[]],0)     # [state, path]

    Visited = [] # The list of the visited state (graph search)

    while not To_visit.isEmpty() :
        position = To_visit.pop()
        # Even if we check before adding the state to the fringe if it has been visited
        # We check again whe we take it form it in case it has been visited since
        while position[0] in Visited:
            position = To_visit.pop() 
        cost = problem.getCostOfActions(position[1])

        if problem.isGoalState(position[0]) : # If the state is the solution we return the path
            return(position[1])
        else : # In the other case we expand the node and add non visited neighbors to the fringe
            Successors = problem.getSuccessors(position[0])
            for neighbor in Successors :
                if not(neighbor[0] in Visited):
                    To_visit.push([neighbor[0], position[1] + [neighbor[1]]],cost + neighbor[2])

            Visited = Visited+[position[0]] # We add the expanded state to the list of the visited ones

    # If the search do not find any solution the algorithm return None
    return None


def nullHeuristic(state:GameState, problem:SearchProblem=None)->List[Direction]:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem:SearchProblem, heuristic=nullHeuristic)->List[Direction]:
    """Search the node that has the lowest combined cost and heuristic first."""

    #####################################################################################
    # We use the same code as before but replacing adding the heuristic to the priority # 
    #                                        cost                                       #
    #####################################################################################

    start = problem.getStartState() # We get the initial state

    To_visit = util.PriorityQueue()       # Our fringe keeping the state to visit under the form:
    To_visit.push([start,[]],heuristic(start,problem))     # [state, path]

    Visited = [] # The list of the visited state (graph search)

    while not To_visit.isEmpty() :
        position = To_visit.pop()
        # Even if we check before adding the state to the fringe if it has been visited
        # We check again whe we take it form it in case it has been visited since
        while position[0] in Visited:
            position = To_visit.pop() 
        cost = problem.getCostOfActions(position[1])

        if problem.isGoalState(position[0]) : # If the state is the solution we return the path
            return(position[1])
        else : # In the other case we expand the node and add non visited neighbors to the fringe
            Successors = problem.getSuccessors(position[0])
            for neighbor in Successors :
                if not(neighbor[0] in Visited):
                    prediction = heuristic(neighbor[0],problem)
                    To_visit.push([neighbor[0], position[1] + [neighbor[1]]], cost + neighbor[2] + prediction)

            Visited = Visited + [position[0]] # We add the expanded state to the list of the visited ones

    # If the search do not find any solution the algorithm return None
    return None


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
