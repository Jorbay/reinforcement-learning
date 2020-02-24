import matplotlib.pyplot as plt

class Plotter():

    def __init__(self):
        self.variables = []
        self.variable_names = []

    def add_variable(self, variable, name):
        self.variables.append(variable)
        self.variable_names.append(name)

    def plot(self):
        number_of_variables = len(self.variables)
        fig, axs = plt.subplots(number_of_variables)
        fig.suptitle('variables over # of trajectories')

        for i in range (0, number_of_variables):
            axs[i].plot(self.variables[i])
            axs[i].set_ylabel(self.variable_names[i])

        plt.show()





