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

    initialState = problem.getStartState() # Initial State

    fringe = util.Stack() # The Fringe
    fringe.push(initialState) # The first state is pushed in the fringe. 
                                            

    visitedStates = set() # The set of the visited states (graph research)
    visitedSuccessors = {initialState: [initialState,None]} # The set of the visited states (graph research) stocked under the form: [previousState, [state, directionTaken]]
    while(not(fringe.isEmpty())):
        state = fringe.pop()
        visitedStates.add(state)

        if (problem.isGoalState(state)): 
            break

        for successor,direction,cost in problem.getSuccessors(state):
            if not(successor in visitedStates):
                visitedSuccessors[successor] = [state,direction]
                
                fringe.push(successor)

    solution = []
    while(state != problem.getStartState()):
        state, direction = visitedSuccessors.get(state)
        
        solution.append(direction)

    solution.reverse()
    return solution


def breadthFirstSearch(problem:SearchProblem)->List[Direction]:
    """Search the shallowest nodes in the search tree first."""


    initialState = problem.getStartState() # Initial State

    fringe = util.Queue() # The Fringe
    fringe.push(initialState) # The first state is pushed in the fringe. 
                                            

    visitedStates = set() # The set of the visited states (graph research)
    visitedSuccessors = {initialState: [initialState,None]} # The set of the visited states (graph research) stocked under the form: [previousState, [state, directionTaken]]
    while(not(fringe.isEmpty())):
        state = fringe.pop()
        if state in visitedStates: 
            continue
        visitedStates.add(state)

        if (problem.isGoalState(state)): 
            break

        for successor,direction,cost in problem.getSuccessors(state):
            if not(successor in visitedStates):
                visitedSuccessors.setdefault(successor,[state,direction])
                
                fringe.push(successor)

    solution = []
    while(state != problem.getStartState()):
        state, direction = visitedSuccessors.get(state)
        
        solution.append(direction)

    solution.reverse()
    return solution

def uniformCostSearch(problem:SearchProblem)->List[Direction]:
    """Search the node of least total cost first."""


    initialState = problem.getStartState() # Initial State

    fringe = util.PriorityQueue() # The Fringe
    fringe.push(initialState, 0) # The first state is pushed in the fringe. 
                                            

    visitedStates = set() # The set of the visited states (graph research)
    visitedSuccessors = {initialState: [initialState,None,0]} # The set of the visited states (graph research) stocked under the form: 
                                                            # [previousState, [state, directionTaken,costOfPath]]
    while(not(fringe.isEmpty())):
        state = fringe.pop()
        if state in visitedStates: 
            continue
        visitedStates.add(state)

        if (problem.isGoalState(state)): 
            break

        for successor,direction,cost in problem.getSuccessors(state):
            if not(successor in visitedStates):
                _,_,costOfPath = visitedSuccessors.get(state)
                newCostOfPath = cost + costOfPath

                if visitedSuccessors.get(successor) and newCostOfPath < visitedSuccessors.get(successor)[2]:
                    visitedSuccessors[successor] = [state,direction,newCostOfPath]
                visitedSuccessors.setdefault(successor,[state,direction,newCostOfPath])

                fringe.push(successor,newCostOfPath)

    solution = []
    while(state != problem.getStartState()):
        state, direction,_ = visitedSuccessors.get(state)
        
        solution.append(direction)

    solution.reverse()
    return solution


def nullHeuristic(state:GameState, problem:SearchProblem=None)->List[Direction]:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem:SearchProblem, heuristic=nullHeuristic)->List[Direction]:
    """Search the node that has the lowest combined cost and heuristic first."""

    initialState = problem.getStartState() # Initial State

    fringe = util.PriorityQueue() # The Fringe
    fringe.push(initialState, 0) # The first state is pushed in the fringe. 
                                            

    visitedStates = set() # The set of the visited states (graph research)
    visitedSuccessors = {initialState: [initialState,None,0]} # The set of the visited states (graph research) stocked under the form: 
                                                            # [previousState, [state, directionTaken,costOfPath]]
    while(not(fringe.isEmpty())):
        state = fringe.pop()
        if state in visitedStates: 
            continue
        visitedStates.add(state)

        if (problem.isGoalState(state)): 
            break

        for successor,direction,cost in problem.getSuccessors(state):
            if not(successor in visitedStates):
                _,_,costOfPath = visitedSuccessors.get(state)
                newCostOfPath = cost + costOfPath + heuristic(state,problem)

                if visitedSuccessors.get(successor) and newCostOfPath < visitedSuccessors.get(successor)[2]:
                    visitedSuccessors[successor] = [state,direction,newCostOfPath]
                visitedSuccessors.setdefault(successor,[state,direction,newCostOfPath])

                fringe.push(successor,newCostOfPath)

    solution = []
    while(state != problem.getStartState()):
        state, direction,_ = visitedSuccessors.get(state)
        
        solution.append(direction)
    solution.reverse()

    print(solution)
    return solution


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
