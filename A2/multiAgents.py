# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # check the manhattanDistance of ghost and pacman, make sure it is not too close
        for ghost in newGhostStates:
            distance = manhattanDistance(ghost.getPosition(), newPos)
            # if it is too close, then this action is bad
            if distance < 2:
                return float("-inf")

        # find the min manhattanDistance between pacman and food
        manh_food = float("inf")
        for i in range(newFood.width):
            for j in range(newFood.height):
                if newFood[i][j]:
                    dis = manhattanDistance((i, j), newPos)
                    if dis < manh_food:
                        manh_food = dis

        # I want the manhattan distance has more effect when it is small
        food_effect = 10/manh_food

        return successorGameState.getScore() + food_effect

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        best_move, value = self.miniMax(gameState, self.index, 0)
        return best_move

    def miniMax(self, gameState, player, depth):
        best_move = None
        value = None

        # check if game is terminal
        if depth >= self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
            return best_move, self.evaluationFunction(gameState)

        # check if player is pacman
        if player == 0:
            value = float("-inf")
        else:
            value = float("inf")

        for move in gameState.getLegalActions(player):
            nxt_state = gameState.generateSuccessor(player, move)
            nxt_move, nxt_val = self.miniMax(nxt_state, (player + 1) % gameState.getNumAgents(), depth + 1)
            if player == 0 and value < nxt_val:
                value, best_move = nxt_val, move
            if player > 0 and value > nxt_val:
                value, best_move = nxt_val, move

        return best_move, value

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        best_move, value = self.AlphaBeta(gameState, self.index, 0, float("-inf"), float("inf"))
        return best_move

    def AlphaBeta(self, gameState, player, depth, alpha, beta):
        best_move = None
        value = None

        # check if game is terminal
        if depth >= self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
            return best_move, self.evaluationFunction(gameState)

        # check if player is pacman
        if player == 0:
            value = float("-inf")
        else:
            value = float("inf")

        for move in gameState.getLegalActions(player):
            nxt_state = gameState.generateSuccessor(player, move)
            nxt_move, nxt_val = self.AlphaBeta(nxt_state, (player + 1) % gameState.getNumAgents(), depth + 1, alpha, beta)
            if player == 0:
                if value < nxt_val:
                    value, best_move = nxt_val, move
                if value >= beta:
                    return best_move, value
                alpha = max(alpha, value)
            if player > 0:
                if value > nxt_val:
                    value, best_move = nxt_val, move
                if value <= alpha:
                    return best_move, value
                beta = min(beta, value)

        return best_move, value

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        best_move, value = self.Expectimax(gameState, self.index, 0)
        return best_move

    def Expectimax(self, gameState, player, depth):
        best_move = None
        value = None

        # check if game is terminal
        if depth >= self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
            return best_move, self.evaluationFunction(gameState)

        # check if player is pacman
        if player == 0:
            value = float("-inf")
        else: # chance
            value = 0.0

        for move in gameState.getLegalActions(player):
            # calculate the probability of this move, assume each move has same probability
            prob_value = 1 / float(len(gameState.getLegalActions(player)))

            nxt_state = gameState.generateSuccessor(player, move)
            nxt_move, nxt_val = self.Expectimax(nxt_state, (player + 1) % gameState.getNumAgents(), depth + 1)
            if player == 0 and value < nxt_val:
                value, best_move = nxt_val, move
            if player > 0:
                value = value + prob_value * nxt_val

        return best_move, value

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
      I use the same way in question 1 to find the min manhattan distance for clostest food (food_effect)
      and avoid ghost.
      Now I want pacman to consider about eating pellet and the ghost.
      Therefore, the min manhattan distance of pellet will be useful. Also, if the pacman can eat the ghost, it is
      better to eat ghost instead of food at that time.
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # check the manhattanDistance of ghost and pacman, make sure it is not too close
    for i in range(len(newGhostStates)):
        distance = manhattanDistance(newGhostStates[i].getPosition(), newPos)
        # if it is too close, then this action is bad
        if distance < 2:
            return float("-inf")

    # find the min manhattanDistance between pacman and food
    sum_dis = 0
    manh_food = float("inf")
    for i in range(newFood.width):
        for j in range(newFood.height):
            if newFood[i][j]:
                dis = manhattanDistance((i, j), newPos)
                sum_dis += dis
                if dis < manh_food:
                    manh_food = dis
    food_effect = 4 / manh_food

    # find the min manhattanDistance between pacman and pellet
    manh_pellet = float("inf")
    for item in currentGameState.getCapsules():
        dis_p = manhattanDistance(item, newPos)
        if dis_p < manh_pellet:
            manh_pellet = dis_p
    pellet_effect = 50 / manh_pellet

    # check if there is a ghost that pacman can eat
    eat_ghost_effect = 0
    for index in range(len(newScaredTimes)):
        dis_t = manhattanDistance(newPos, newGhostStates[index].getPosition())
        if newScaredTimes[index] > 0 and dis_t < newScaredTimes[index]:
            eat_ghost_effect = 1000 / dis_t

    return currentGameState.getScore() + food_effect + pellet_effect + eat_ghost_effect

# Abbreviation
better = betterEvaluationFunction

