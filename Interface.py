import numpy as np
import logging

from Configs import Configs
from Experiment import Experiment


def main():
    # Logging levels:
    # Error - items that cause exceptions
    # Warning - items that may (but not necessarily) indicate a problem with the simulation
    # Info - items that are important to track for the experiment (e.g. runTime, alpha, etc.)
    # Debug - items useful for debugging
    logging.basicConfig(filename='sim.log',
                        level=logging.WARNING,
                        format='%(levelname)s:%(message)s',
                        filemode='w')  # Will write over the previous log. 'a' will append
    logging.info('Starting Log...')

    """
    Input Baseline experiment parameters here
    """
    savePath = "Completed Experiments"
    startingExperimentNumber = -99
    numIterations = 10  # (default = 1000)
    runTime = 500  # (default = 500)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    alphaOption = "Varied"  # default
    # alphaOption = 'Uniform'
    alphaValue = 0.1  # (required only if alphaOption = 'Uniform'; default = 0.1)
    alphaValueList = [0.1, 0.1, 0.05, 0.15, 0.1]  # (Default = [0.1, 0.1, 0.05, 0.15, 0.1])
    # alphaValueList = [0.2, 0, 0, 0, 0]
    demandMu = 100  # (default =  100)
    demandStd = 1  # (default = 1)
    historyTime = 10  # (default = 10)
    shipDelayOption = 'Uniform'  # (default = 'Uniform')
    shipDelayList = None  # (required only if shipDelayOption  = 'Varied'; default = None)
    shipDelayValue = 2  # (default = 2)
    shocks = np.array([[10, 1], [200, -.8]])  # (default = [[50, 1], [250, -.8]]); OR ([[200, 1]]) OR ([[200, -.8]])
    smoothingValue = 2  # (default = 2)
    # wholesalerProfileOption = 'Optimize'
    wholesalerProfileOption = 'Default'
    wholesalerProfile = [0.5, 0.5]  # (default =  [0.5, 0.5])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # The following are only required for optimization experiments
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    covarianceMatrix = np.array([[1, 0],
                                 [0, 1]
                                 ])
    historicalReturnRuleOption = 2  # (default = 3)
    numTimePeriodsForCalc = 5  # (default = 5)
    riskTolerance = 0.2  # (default = 0.5)
    """
    No further input needed
    """
    configs1 = Configs(runTime=runTime,
                       alphaOption=alphaOption,
                       alphaValueList=alphaValueList,
                       alphaValue=alphaValue,
                       demandMu=demandMu,
                       demandStd=demandStd,
                       historyTime=historyTime,
                       numIterations=numIterations,
                       shipDelayOption=shipDelayOption,
                       shipDelayList=shipDelayList,
                       shipDelayValue=shipDelayValue,
                       shocks=shocks,
                       smoothingValue=smoothingValue,
                       wholesalerProfileOption=wholesalerProfileOption,
                       wholesalerProfile=wholesalerProfile,
                       covarianceMatrix=covarianceMatrix,
                       historicalReturnRuleOption=historicalReturnRuleOption,
                       numTimePeriodsForCalc=numTimePeriodsForCalc,
                       riskTolerance=riskTolerance,
                       )
    experiment1 = Experiment(savePath, configs1, startingExperimentNumber=startingExperimentNumber)
    experiment1.setup()
    experiment1.run()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    """
    Input changed parameters here
    """
    configs2 = Configs(runTime=runTime,
                       alphaOption=alphaOption,
                       alphaValueList=alphaValueList,
                       alphaValue=alphaValue,
                       demandMu=demandMu,
                       demandStd=demandStd,
                       historyTime=historyTime,
                       numIterations=numIterations,
                       shipDelayOption=shipDelayOption,
                       shipDelayList=shipDelayList,
                       shipDelayValue=shipDelayValue,
                       shocks=shocks,
                       smoothingValue=smoothingValue,
                       wholesalerProfileOption=wholesalerProfileOption,
                       wholesalerProfile=wholesalerProfile,
                       covarianceMatrix=covarianceMatrix,
                       historicalReturnRuleOption=historicalReturnRuleOption,
                       numTimePeriodsForCalc=numTimePeriodsForCalc,
                       riskTolerance=0.5,
                       )
    """
    No further input needed
    """
    experiment2 = Experiment(savePath, configs2, startingExperimentNumber=startingExperimentNumber)
    experiment2.setup()
    experiment2.run()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    """
    Input changed parameters here
    """
    configs3 = Configs(runTime=runTime,
                       alphaOption=alphaOption,
                       alphaValueList=alphaValueList,
                       alphaValue=alphaValue,
                       demandMu=demandMu,
                       demandStd=demandStd,
                       historyTime=historyTime,
                       numIterations=numIterations,
                       shipDelayOption=shipDelayOption,
                       shipDelayList=shipDelayList,
                       shipDelayValue=shipDelayValue,
                       shocks=shocks,
                       smoothingValue=smoothingValue,
                       wholesalerProfileOption=wholesalerProfileOption,
                       wholesalerProfile=wholesalerProfile,
                       covarianceMatrix=covarianceMatrix,
                       historicalReturnRuleOption=historicalReturnRuleOption,
                       numTimePeriodsForCalc=numTimePeriodsForCalc,
                       riskTolerance=0.8,
                       )
    """
    No further input needed
    """
    experiment3 = Experiment(savePath, configs3, startingExperimentNumber=startingExperimentNumber)
    experiment3.setup()
    experiment3.run()

    logging.info('Ending log.')


if __name__ == '__main__':
    main()