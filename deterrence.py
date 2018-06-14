from random import choice, random
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

numStates = 100             # Number of states in a simulation
turnLength = 100            # How long states play before their scores are measured and mutation happens
numTurns = 10000            # How many cycles of mutation and play
mutateProbability = 0.05    # Probability of choosing a random retaliation probability instead of inheriting one
defenderCost = 0.5          # Cost to the defender of being attacked
retaliationCost = 0.75      # Cost to the defender of being attacked and retaliating
retaliationEffect = 0.5     # Cost to the attacker of being retaliated against

attackTypes = 10
initialValue = 0
intensities = [a + 1 for a in range(attackTypes)]

def mutate(val):
    if random() < mutateProbability:
        return random()
    else:
        return val

class State:
    def __init__(self, name):
        self.name = name
        self.retaliationTable = {intensity: random() for intensity in intensities}
        self.reset()

    def reset(self):
        self.value = initialValue
        self.numIgnores = {intensity: 1 for intensity in intensities}
        self.numRetaliates = {intensity: 1 for intensity in intensities}

    def changeStrategy(self, winner):
        for i in intensities:
            self.retaliationTable[i] = mutate(choice([self.retaliationTable[i], winner.retaliationTable[i]]))

class Context:
    def __init__(self, rationality, attribution):
        self.rationality = rationality
        self.attribution = attribution
        self.states = [State(index) for index in range(numStates)]
        self.data = []
    
    def expectedValue(self, intensity, defender):
        pNo = defender.numIgnores[intensity] / (defender.numIgnores[intensity] + defender.numRetaliates[intensity])
        pSuccess = (1 - pNo) * self.attribution
        pFail = (1 - pNo) * (1 - self.attribution)
        return intensity * (pNo + pFail - pSuccess * retaliationEffect)
        
    def update(self, ax):
        for state in self.states:
            state.reset()
    
        for tick in range(turnLength):
            for attacker in self.states:
                while True:
                    defender = choice(self.states)
                    if attacker is not defender:
                        break

                if random() > self.rationality:
                    intensity = choice(intensities)
                else:
                    intensity = max(intensities, key=lambda intensity: self.expectedValue(intensity, defender))
                if random() > defender.retaliationTable[intensity]:
                    # No retaliation
                    attacker.value += intensity
                    defender.value -= intensity  * defenderCost
                    defender.numIgnores[intensity] += 1
                elif random() > self.attribution:
                    # Failed retaliation
                    attacker.value += intensity
                    defender.value -= intensity * retaliationCost
                    defender.numRetaliates[intensity] += 1
                else:
                    # Successful retaliation
                    attacker.value -= intensity * retaliationEffect
                    defender.value -= intensity * retaliationCost
                    defender.numRetaliates[intensity] += 1

        loser = min(self.states, key=lambda state: state.value)
        winner = max(self.states, key=lambda state: state.value)
        loser.changeStrategy(winner)
        strategyTable = np.array([[state.retaliationTable[i] for i in intensities] for state in self.states])
        self.data.append(np.mean(strategyTable, axis=0))
        
        if ax:
            data = np.array(self.data)
            ax.clear()
            for index, intensity in enumerate(intensities):
                ax.plot(data[:,index], color=(index/(len(intensities)-1),0,0), label=str(intensity))
            if len(self.data) >= 2000:
                ax.set_xlim(len(self.data) - 2000, len(self.data) - 1)
            else:
                ax.set_xlim(0, 1999)
            ax.set_ylim(0, 1)
            ax.legend()
        return ax

def update(time, context, ax, fig):
    print(time)
    ax = context.update(ax)
    if fig:
        fig.suptitle('Step: {}'.format(time))
    return ax
    
def runSimulation(rationality, attribution, show, save):
    context = Context(rationality, attribution)
    if show or save:
        fig, ax = plt.gcf(), plt.gca()
        ani = animation.FuncAnimation(fig, update, numTurns, fargs=(context, ax, fig), interval=10, blit=False, repeat=False)
        if show:
            plt.show()
        if save:
            ani.save('{}.mp4'.format('deterrence'), fps=10)
        plt.close()
    else:
        for turn in range(numTurns):
            update(turn, context, None, None)
    return context
    
# This figure aggregates data from multiple runs. As such, it takes a long time to run, so the process of making it is split in two:
# The first step is to generate the data and save it as a *.csv file
# The second step is to generate the figure from the data (so you can experiment with different formatting without rerunning the simulation)
    
def make_figure_data():
    data = []
    for attribution in np.linspace(0.0, 1.0, 11):
        for rationality in np.linspace(0.0, 1.0, 11):
            for repeat in range(10):
                context = runSimulation(rationality, attribution, False, False)
                last = context.data[-1]
                row = {'Rationality': rationality, 'Attribution': attribution}
                for index, value in enumerate(last):
                    row[intensities[index]] = value
                data.append(row)
    data = pd.DataFrame(data)
    data.to_csv('Full.csv', index=False)
    maxes = data.apply(lambda row: max(d for l,d in row.iteritems() if l != 'Rationality' and l != 'Attribution'), axis=1, reduce=True)
    data = data[['Rationality', 'Attribution']]
    data['Max Deterrence'] = maxes
    data = data.groupby(['Rationality', 'Attribution']).mean().reset_index()
    data.to_csv('Summary.csv', index=False) 
    
def make_figure():
    from mpl_toolkits.mplot3d import Axes3D
    data = pd.read_csv('Summary.csv')
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(data.loc[:,'Rationality'], data.loc[:,'Attribution'], data.loc[:,'Max Deterrence'])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel('Rationality')
    ax.set_ylabel('Attribution')
    ax.set_zlabel('Max Deterrence')
    plt.show()
    plt.close()

if __name__ == "__main__":
    runSimulation(0.9, 0.9, True, False)