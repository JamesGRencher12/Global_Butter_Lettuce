
# Mapping implements __getitem__, which allows for indexing experimentCharts[2] would grab the second chart, for example
from collections.abc import Mapping
import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime

import os
# Find the root directory to save the path for the chart
abspath = os.path.abspath(__file__)
rootDir = os.path.dirname(abspath)


class ExperimentData(Mapping):
    """
    A class which handles data storage, processing, analysis. and presentation for the simulation.

    A single instantiation of the data contains all the data from one experiment. An experiment is defined as a unique
    collection of simulation parameters. A single experiment will have n simulations.



    Instance Attributes
    -------------------


    Instance Methods
    ----------------

    """

    def __init__(self,
                 numAgents,
                 totalTime,
                 experimentNumber,
                 savePath):
        self.numAgents = numAgents
        self.totalTime = totalTime
        self.experimentNumber = experimentNumber

        self.averageData = np.zeros((self.numAgents,
                                     self.totalTime), dtype=int)

        saveFolder = os.path.join(rootDir, savePath)
        if not os.path.exists(saveFolder):
            os.makedirs(saveFolder)
        os.chdir(saveFolder)  # Change the working directory to rootDir/savePath

    def __getitem__(self, item):
        return self.__dict__[item]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def writeCsvData(self, totalTime, timeStart, now):
        """


        :return:
        """
        xAxis = list(range(timeStart, totalTime))  # Time axis
        header = np.array(xAxis)
        writtenData = np.vstack([header, self.averageData[:, timeStart::]]).astype(int)
        fileName = f"Experiment_{self.experimentNumber}_{now}.csv"
        key = np.vstack([["Time"], ["Firm 0"], ["Firm 1"], ["Firm 2"], ["Firm 3"], ["Firm 4"]])
        writtenData = np.hstack([key, writtenData])

        with open(fileName, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(writtenData)


class ExperimentCharts(Mapping):
    """
    Container for all the graphical representations of the simulation
    """

    def __init__(self,
                 experimentData):
        self.experimentData = experimentData
        self.charts = []

    # Indexing here could be meaningful, think carefully about how to do this.
    def __getitem__(self, item):
        return self.__dict__[item]

    # Should iterate through the charts only
    def __iter__(self):
        return iter(self.__dict__)

    # Should return the number of charts
    def __len__(self):
        return len(self.__dict__)

    def plotAverageData(self, numSims, totalTime, timeStart, optimized, now):
        # Demand Data
        plt.figure(1)
        if optimized == 'Optimize':
            chartInfo = 'Optimized'
        else:
            chartInfo = "Baseline"
        chartTitle = f"{chartInfo} Demand Chart, averaged over {numSims} sims"
        plt.title(chartTitle)
        plt.xlabel('Time')
        plt.ylabel('Demand')
        xAxis = list(range(timeStart, totalTime))  # Time axis
        for firm in range(len(self.experimentData.averageData)):
            data = self.experimentData.averageData[firm, timeStart::]
            label = f"Firm {firm}"
            plt.plot(xAxis, data, label=label)
        plt.legend()

        def saveChart():
            fileName = f"Experiment_{self.experimentData.experimentNumber}_{now}.png"
            plt.savefig(fileName)

        saveChart()
        plt.show()


class ExperimentPerformance:
    """
    Each ExperimentPerformance object contains data on time performance of the overall experiment
    """
    pass


class ExperimentLog:
    """
    Each ExperimentLog object contains a log file for all the simulations.
    """
