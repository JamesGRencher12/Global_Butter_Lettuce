
import logging
from collections.abc import Mapping
import itertools
import numpy as np
from datetime import datetime

from Configs import Configs
from Simulation import Simulation
import Firm
import ExperimentData


class Experiment(Mapping):
    """
    A class which handles the parameters and data output from the simulations.

    A single instantiation of an Experiment object has:
        An instantiation of the Configs object which stores the experiment parameters.
        An instantiation of the ExperimentData object which stores the data from all n simulations using Configs
            parameters.
        n instantiations of the simulation object which runs the dynamics for one iteration.

    The user will input the parameters, which is then put into a configuration object. That is passed to the experiment
    object. The experiment object will run a set number of simulations and save that information.

    Class Attributes
    ----------------
    idIter: iterator
        Creates a new iterator that will be used to create a unique identifier for the experiment. Will increment
        each time a new Experiment object is created.

    Instance Attributes
    -------------------
    experimentConfigs: Configs.Configs
        A Configs object which stores all the parameters used for the experiment.
    firmList: List
        A list of Firm3 objects representing the firms in the simulation.
    id: int
        Unique identifier for the experiment
    savePath: str
        A string representing the local place where the packaged experiment data will be saved to.
    startingExperimentNumber: int
        Optional argument. If prior experiments have been performed, the id will begin with this value and increment
        from there.

    Instance Methods
    ----------------

    """
    idIter = itertools.count()  # Class variable

    def __init__(self,
                 savePath,
                 experimentConfigs,
                 startingExperimentNumber=1):
        self.id = next(self.idIter) + startingExperimentNumber  # The experiment number

        self.experimentConfigs = experimentConfigs
        if not isinstance(self.experimentConfigs, Configs):
            message = f"Experiment: {self.id}, non-Config object input into experimentConfigs"
            logging.error(message)
            raise TypeError(message)

        self.savePath = savePath
        if not isinstance(self.savePath, str):
            message = f"Experiment: {self.id}, non-string input into savePath"
            logging.error(message)
            raise TypeError(message)

        self.firmList = []

        self.experimentData = ExperimentData.ExperimentData(self.experimentConfigs.numAgents,
                                                            self.experimentConfigs.historyTime +
                                                            self.experimentConfigs.runTime,
                                                            self.id,
                                                            self.savePath)
        self.experimentCharts = None

        self.now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    def __getitem__(self, item):
        return self.__dict__[item]

    # Iterate through each simulation?
    def __iter__(self):
        return iter(self.__dict__)

    # Define the length of "Experiment" as the number of simulations (n)?
    def __len__(self):
        return len(self.__dict__)

    def createCharts(self):
        """
        After the simulations have run, the method will then create the charts

        :return:
        """
        self.experimentCharts = ExperimentData.ExperimentCharts(self.experimentData)
        self.experimentCharts.plotAverageData(self.experimentConfigs.numIterations,
                                              self.experimentConfigs.historyTime + self.experimentConfigs.runTime,
                                              self.experimentConfigs.historyTime,
                                              self.experimentConfigs.wholesalerProfileOption,
                                              self.now)

    def packageData(self):
        """
        Will package together the .csv, .txt, .log, and .png files for exporting:
            1. A .txt file that contains the parameters for the experiment (related to Configs.Configs)
            2. A .csv file that contains the raw data for the experiment (related to ExperimentData)
            3. A .png file(s) that contain(s) the output charts (related to ExperimentChart)
            4. A .log file that contains the logging information for all n simulations

        :return:
        """
        pass

    def run(self):
        """
        Run the experiment

        :return:
        """

        for i in range(self.experimentConfigs.numIterations):
            sim = Simulation()
            sim.initializeSim(testOption=False)
            sim.runSimulation()
            data = sim.processData(i)
            data = np.array(data)
            self.experimentData.averageData = np.add(self.experimentData.averageData, data)
            message = f"Experiment {self.id} Sim {i} running..."
            print(message)
            sim.resetFirms()
        self.experimentData.averageData = self.experimentData.averageData / self.experimentConfigs.numIterations
        self.createCharts()
        self.experimentData.writeCsvData(self.experimentConfigs.historyTime + self.experimentConfigs.runTime,
                                         self.experimentConfigs.historyTime,
                                         self.now)
        self.writeTxtData()

    def savePackagedData(self):
        """
        Will write the packaged data

        :return:
        """
        pass

    def setup(self):
        """
        Set up the experiment to be run

        :return:
        """

        def createFirms():
            """
            Establishes five firms for the simulation. One retailer, one wholesaler, two manufacturers, and one
            raw materials producer.  Puts the firms in a list and returns the list

            :return:
                List of Firm3 objects
            """
            f0 = Firm.Retailer(alpha=self.experimentConfigs.alphaValueList[0],
                                runTime=self.experimentConfigs.runTime,
                                historyTime=self.experimentConfigs.historyTime,
                                shipDelay=self.experimentConfigs.shipDelayList[0],
                                idNum=0,
                                supplierList=[1],
                                customerList=[-99])
            f1 = Firm.Wholesaler(alpha=self.experimentConfigs.alphaValueList[1],
                                  runTime=self.experimentConfigs.runTime,
                                  historyTime=self.experimentConfigs.historyTime,
                                  shipDelay=self.experimentConfigs.shipDelayList[1],
                                  idNum=1,
                                  supplierList=[2, 3],
                                  customerList=[0],
                                  wholesalerProfile=self.experimentConfigs.wholesalerProfile,
                                  wholesalerProfileOption=self.experimentConfigs.wholesalerProfileOption,
                                  numTimePeriodsForCalc=self.experimentConfigs.numTimePeriodsForCalc,
                                  historicalReturnRuleOption=self.experimentConfigs.historicalReturnRuleOption,
                                  covarianceMatrix=self.experimentConfigs.covarianceMatrix,
                                  riskTolerance=self.experimentConfigs.riskTolerance)
            f2 = Firm.Manufacturer(alpha=self.experimentConfigs.alphaValueList[2],
                                    runTime=self.experimentConfigs.runTime,
                                    historyTime=self.experimentConfigs.historyTime,
                                    shipDelay=self.experimentConfigs.shipDelayList[2],
                                    idNum=2,
                                    supplierList=[4],
                                    customerList=[1])
            f3 = Firm.Manufacturer(alpha=self.experimentConfigs.alphaValueList[3],
                                    runTime=self.experimentConfigs.runTime,
                                    historyTime=self.experimentConfigs.historyTime,
                                    shipDelay=self.experimentConfigs.shipDelayList[3],
                                    idNum=3,
                                    supplierList=[4],
                                    customerList=[1])
            f4 = Firm.RawMaterials(alpha=self.experimentConfigs.alphaValueList[4],
                                    runTime=self.experimentConfigs.runTime,
                                    historyTime=self.experimentConfigs.historyTime,
                                    shipDelay=self.experimentConfigs.shipDelayList[4],
                                    idNum=4,
                                    supplierList=[-99],
                                    customerList=[2, 3])
            firmList = [f0, f1, f2, f3, f4]
            return firmList

        self.firmList = createFirms()
        # The class is initialized first to save on runTime. Values do not change from sim to sim within the experiment
        Simulation.initializeClass(self.experimentConfigs, self.firmList)

    def writeTxtData(self):
        """


        :return:
        """
        lines = [f"Experiment_{self.id}",
                 f"runTime= {self.experimentConfigs.runTime}",
                 f"historyTime= {self.experimentConfigs.historyTime}",
                 f"numIterations= {self.experimentConfigs.numIterations}",
                 f"alphaValueList= {self.experimentConfigs.alphaValueList}",
                 f"demandMu= {self.experimentConfigs.demandMu}",
                 f"demandStd= {self.experimentConfigs.demandStd}",
                 f"shipDelayList= {self.experimentConfigs.shipDelayList}",
                 f"shocks=\n {self.experimentConfigs.shocks}",
                 f"smoothingValue= {self.experimentConfigs.smoothingValue}",
                 f"wholesalerProfile= {self.experimentConfigs.wholesalerProfile}",
                 f"covarianceMatrix=\n {self.experimentConfigs.covarianceMatrix}",
                 f"historicalReturnRuleOption= {self.experimentConfigs.historicalReturnRuleOption}",
                 f"numTimePeriodsForCalc= {self.experimentConfigs.numTimePeriodsForCalc}",
                 f"riskTolerance= {self.experimentConfigs.riskTolerance}"]
        fileName = f"Experiment_{self.id}_{self.now}.txt"
        with open(fileName, 'w') as f:
            for line in lines:
                f.write(line)
                f.write("\n")
