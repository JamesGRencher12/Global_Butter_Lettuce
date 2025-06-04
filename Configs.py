import logging
from collections.abc import Mapping
import numpy as np


class Configs(Mapping):
    """
    A class for establishing simulation configurations. To be used in conjunction with Simulation.py

    Instance Attributes
    ----------
    alphaOption: str
        Takes values 'Uniform' or 'Varied'. If varied is chosen, each firm has different alpha values and needs to be
        specified in alphaValueList. If uniform is chosen, all firms have the same alpha value which needs to be
        specified with alphaValue.
    alphaValue: float
        Need only if 'Uniform' is chosen for alphaOption. A value representing the percent of inventory that is
        maintained as safety stock each time period.
    alphaValueList: list
        Need only if 'Varied' is chosen for alphaOption. A list of values where each entry represents the percent
        of inventory that is maintained as safety stock each time period for that specific firm.
    covarianceMatrix: np array
        A 2x2 array representing the covariance in return between firm 2 and 3, both suppliers for firm 1
    demandMu: int
        Value representing the average end consumer demand (i.e. the demand signal to firm 0). Every time period a
        random draw from a normal distribution is made with mu=demandMu and variance=demandStd. In other words,
        end consumer demand is a random variable ~N(demandMu, demandStd)
    demandStd: int
        Value representing the standard deviation in end consumer demand (i.e. the demand signal to firm 0). Every time
        period a random draw from a normal distribution is made with mu=demandMu and standard deviation=demandStd. In
        other words, end consumer demand is a random variable ~N(demandMu, demandStd)
    graphName: str
        Currently only option is 'Kite'
    historyTime: int
        Number of historical time periods to create before simulation runs. Used for some calculations.
    historicalReturnRuleOption: int
        Historical return is the average percentage of orders by a firm that was fulfilled. When the wholesaler portfolio
        is 0 for a given firm, it is unclear how the historicalReturn calculation should treat it. As a 0? a 1?
        Option 1: Treated as a 0
        Option 2: Treated as a 1
        Option 3 (Default): To avoid such a decision, choosing this option will mean that the wholesaler uses the last n
            timeperiods where an order was sent to that supplier. i.e., It will ignore time periods where order was 0.
    numAgents:
        Parameter value for the number of agents in the initialized simulation
    numIterations: int
        The number of times the simulation will be repeated with the exact same parameter values
    numTimePeriodsForCalc: int
        The number of previous time periods that the wholesaler will use to calculate historical return, which is
        used to optimize their portfolio
    riskTolerance: float
        A parameter used in the optimization calculation.  A higher value will place more emphasis on rewards at
        a higher risk, whereas lower values emphasize reducing risk at the expense of lower rewards. It is equivalent
        to desired return, therefore must take on values of [0,1].
    runTime: int
        The amount of time periods the simulation will run (not including historical time)
    shipDelayList: list
        A list where each entry is a value representing an individual firm's shipping delay value. Only required if
        shipDelayOption =  'Varied'
    shipDelayOption: str
        Takes value of 'Uniform' or 'Varied'. Uniform means all firms in the simulation have the same shipDelayValue.
    shipDelayValue: int
        The number of time periods it takes for product to be received. Ordered product is received in the time period
        (t + shipDelayValue). An important implication is that there will be (shipDelayValue - 1) time periods of sales
        while the product is en route.
        Example: (t=10, shipDelayValue=2)
            [t=10] Product ordered at end of time 10
            [t=11] sales actualized for time 11
            [t=12] Product received at beginning of time 12
    shocks: NumPy 2D Array
        An array containing the time periods (column 1) and shock value (column 2; as a percentage change)

        Example: [[10, 0.2][20, -0.3]] Has a shock of 20% increase in time period 10 and a shock of 30% down in time
        period 20. If the demand value for both time periods was originally 10, the new value would be 1.2 and 0.7
        respectively
    smoothingValue: int
        smoothingValue takes a value of 1 or 2.
        Choosing 1 turns off smoothing (future demand = current time period demand).
        Choosing 2 turns on smoothing of 1 time period
            [future demand = (current demand + current forecasted demand) / 2]
    wholesalerProfile: list
        A list where each entry is a float representing the percentage of supplies that will be ordered from a
        particular supplier. Must sum to 1.

        Currently the simulation only allows two suppliers for the wholesaler. Thus, index 0
        is the percentage firm 1 (wholesaler) buys from firm 2, and index 1 is the percentage firm 1 buys from firm
        3.

        In the special case where wholesalerProfileOption = 'Optimized', this profile changes
        every time period. Otherwise, it stays constant the entire simulation.
    wholesalerProfileOption: str
        Takes value of 'Optimize' or 'Default'. If optimize is chosen, every time period the wholesaler will try to
        optimize their supplier portfolio. Otherwise, the wholesaler will never change their portfolio.

    Instance Methods
    ---------

    """

    def __init__(self,
                 runTime=500,
                 alphaOption='Uniform',
                 alphaValue=0.1,
                 alphaValueList=None,
                 demandMu=100, #This is how many thousands of pounds of butter lettuce are brought into the US annually
                 demandStd=1,
                 historyTime=10,
                 numIterations=100,
                 shipDelayOption='Uniform',
                 shipDelayList=None,
                 shipDelayValue=2,
                 shocks=None,
                 smoothingValue=2,
                 wholesalerProfileOption='Default',
                 wholesalerProfile=None,
                 covarianceMatrix=None,
                 historicalReturnRuleOption=3,
                 numTimePeriodsForCalc=5,
                 riskTolerance=0.5,
                 acreage = 500, #This number is completely made up
                 miles_mexico = 700,
                 miles_us = 1360,
                 border_delay = None,
                 shipment_size = None,
                 storage_time = None
                 ):
        self.graphName = 'Kite'  # Cannot change this currently
        self.numAgents = 5  # Cannot change this currently

        self.alphaOption = alphaOption
        logging.info(f"Set alphaOption to {self.alphaOption}")
        self.alphaValue = alphaValue
        logging.info(f"Set alphaValue to {self.alphaValue}")
        if alphaOption == 'Uniform':
            if alphaValue < 0:
                message = "Alpha value cannot be less than 0"
                logging.error(message)
                raise Exception(message)
            self.alphaValueList = [alphaValue, alphaValue, alphaValue, alphaValue, alphaValue]
            logging.info(f"Set alphaValueList to {self.alphaValueList}")
        elif alphaOption == 'Varied':
            if alphaValueList is None:
                message = "alphaValueList must be specified if alphaValueOption=Varied is chosen"
                logging.error(message)
                raise Exception(message)
            if len(alphaValueList) != self.numAgents:
                message = "An alpha value must be input for each firm"
                logging.error(message)
                raise Exception(message)
            for value in alphaValueList:
                if value < 0:
                    message = "Alpha value cannot be less than 0"
                    logging.error(message)
                    raise Exception(message)
            self.alphaValueList = alphaValueList
            logging.info(f"Set alphaValueList to {self.alphaValueList}")
        else:
            message = "Unknown option chosen for alphaOption"
            logging.error(message)
            raise Exception(message)

        self.covarianceMatrix = covarianceMatrix
        logging.info(f"Set covarianceMatrix to {self.covarianceMatrix}")
        if covarianceMatrix is None:
            message = "Covariance matrix not initialized"
            logging.error(message)
            raise Exception(message)
        if not isinstance(covarianceMatrix, np.ndarray):
            message = "Covariance matrix must be an NumpPy array"
            logging.error(message)
            raise Exception(message)
        test1 = covarianceMatrix > 1
        test2 = covarianceMatrix < 0
        if np.sum(test1) + np.sum(test2) != 0:
            message = "Invalid values for covariance matrix. Must be [0, 1]"
            logging.error(message)
            raise Exception(message)

        self.demandMu = demandMu
        logging.info(f"Set demandMu to {self.demandMu}")
        if not isinstance(demandMu, int):
            message = "Mu value must be an int"
            logging.error(message)
            raise TypeError(message)
        if demandMu < 0:
            message = "Mu Value cannot be less than 0"
            logging.error(message)
            raise Exception(message)

        self.demandStd = demandStd
        logging.info(f"Set demandStd to {self.demandStd}")
        if not isinstance(demandStd, int):
            message = "Variance must be an int"
            logging.error(message)
            raise TypeError(message)
        if demandStd < 0:
            message = "Variance cannot be less than 0"
            logging.error(message)
            raise Exception(message)

        self.historicalReturnRuleOption = historicalReturnRuleOption
        logging.info(f"Set historicalReturnRuleOption to {self.historicalReturnRuleOption}")
        if not isinstance(historicalReturnRuleOption, int):
            message = "Historical return rule option must be an integer value"
            logging.error(message)
            raise TypeError(message)
        if historicalReturnRuleOption > 3 or historicalReturnRuleOption < 1:
            message = "Invalid option for historical return rule: must be 1, 2, or 3"
            logging.error(message)
            raise Exception(message)

        self.historyTime = historyTime  # Number of historical time periods. Used for firm calculations
        logging.info(f"Set historyTime to {self.historyTime}")
        if not isinstance(historyTime, int):
            message = "Historical time must be an int"
            logging.error(message)
            raise TypeError(message)
        if historyTime < 0:
            message = "Historical time cannot be a value less than 0"
            logging.error(message)
            raise Exception(message)

        self.numIterations = numIterations
        if not isinstance(self.numIterations, int):
            message = "Number of iterations must be an int"
            logging.error(message)
            raise TypeError(message)
        if self.numIterations <= 0:
            message = "Number of iterations must be 1 or greater"
            logging.error(message)
            raise Exception(message)

        self.numTimePeriodsForCalc = numTimePeriodsForCalc
        logging.info(f"Set numTimePeriodsForCalc to {self.numTimePeriodsForCalc}")
        if not isinstance(numTimePeriodsForCalc, int):
            message = "Number of time periods for return calculation must be an int"
            logging.error(message)
            raise TypeError(message)
        if numTimePeriodsForCalc < 1:
            message = "Number of time periods for return calculation cannot be less than 1"
            logging.error(message)
            raise Exception(message)
        if numTimePeriodsForCalc > historyTime:
            message = "historyTime must be greater than or equal to numTimePeriodForCalc"
            logging.error(message)
            raise Exception(message)

        self.riskTolerance = riskTolerance
        logging.info(f"Set riskTolerance to {self.riskTolerance}")
        if riskTolerance <= 0:
            message = "Risk tolerance value cannot be less than 0"
            logging.error(message)
            raise Exception(message)
        if riskTolerance > 1:
            message = "Risk tolerance value cannot be greater than 1. (Risk tolerance is equivalent to "
            "desired return, and return cannot be greater than 1."
            logging.error(message)
            raise Exception(message)

        self.runTime = runTime  # Simulation run time not including historical time periods
        logging.info(f"Set runTime to {self.runTime}")
        if not isinstance(runTime, int):
            message = "Runtime must be an int"
            logging.error(message)
            raise TypeError(message)
        if runTime < 0:
            message = "Runtime cannot be a value less than 0"
            logging.error(message)
            raise Exception(message)
        
        #These are the variables for the global supply chain.
        self.acreage = acreage
        logging.info(f"Set acreage to {self.acreage}")
        if self.acreage is None:
            logging.warning("Acreage is not set; defaulting to None")
        elif self.acreage <= 0:
            message = "Acreage must be a positive number"
            logging.error(message)
            raise Exception(message)

        self.miles_mexico = miles_mexico
        logging.info(f"Set miles_mexico to {self.miles_mexico}")
        if self.miles_mexico is None:
            logging.warning("Miles from Mexico to border is not set; defaulting to None")
        elif self.miles_mexico < 0:
            message = "Miles from Mexico must be non-negative"
            logging.error(message)
            raise Exception(message)

        self.miles_us = miles_us
        logging.info(f"Set miles_us to {self.miles_us}")
        if self.miles_us is None:
            logging.warning("Miles from border to destination is not set; defaulting to None")
        elif self.miles_us < 0:
            message = "Miles from US border must be non-negative"
            logging.error(message)
            raise Exception(message)

        self.border_delay = border_delay
        logging.info(f"Set border_delay to {self.border_delay}")
        if self.border_delay is None:
            logging.warning("Border delay is not set; defaulting to None")
        elif self.border_delay < 0:
            message = "Border delay must be non-negative"
            logging.error(message)
            raise Exception(message)

        self.shipment_size = shipment_size
        logging.info(f"Set shipment_size to {self.shipment_size}")
        if self.shipment_size is None:
            logging.warning("Shipment size is not set; defaulting to None")
        elif self.shipment_size <= 0:
            message = "Shipment size must be a positive number"
            logging.error(message)
            raise Exception(message)

        self.storage_time = storage_time
        logging.info(f"Set storage_time to {self.storage_time}")
        if self.storage_time is None:
            logging.warning("Storage time is not set; defaulting to None")
        elif self.storage_time < 0:
            message = "Storage time must be non-negative"
            logging.error(message)
            raise Exception(message)


        self.shipDelayOption = shipDelayOption
        logging.info(f"Set shipDelayOption to {self.shipDelayOption}")
        self.shipDelayValue = shipDelayValue
        logging.info(f"Set shipDelayValue to {self.shipDelayValue}")
        if shipDelayOption == 'Uniform':
            if shipDelayValue > historyTime + 1:
                message = "Must have more historical time periods than shipping delay value"
                logging.error(message)
                raise Exception(message)
            if not isinstance(shipDelayValue, int):
                message = "Shipping delay value must be an int"
                logging.error(message)
                raise Exception(message)
            if shipDelayValue < 0:
                message = "Shipping delay values cannot be less than 0"
                logging.error(message)
                raise Exception(message)
            self.shipDelayList = [shipDelayValue, shipDelayValue, shipDelayValue, shipDelayValue, shipDelayValue]
            logging.info(f"Set shipDelayList to {self.shipDelayList}")
        elif shipDelayOption == 'Varied':
            if shipDelayList is None:
                message = "shipDelayList must be specified if shipDelayOption=Varied is chosen"
                logging.error(message)
                raise Exception(message)
            if len(shipDelayList) != self.numAgents:
                message = "A ship delay value must be input for each firm"
                logging.error(message)
                raise Exception(message)
            if max(shipDelayList) > historyTime:
                message = "Must have more historical time periods than maximum shipping delay value"
                logging.error(message)
                raise Exception(message)
            for value in shipDelayList:
                if not isinstance(value, int):
                    message = "All shipping delay values must be int"
                    logging.error(message)
                    raise Exception(message)
                if value < 0:
                    message = "Shipping delay values cannot be less than 0"
                    logging.error(message)
                    raise Exception(message)
        else:
            message = "Unknown option chosen for shipDelayOption"
            logging.error(message)
            raise Exception(message)

        if shocks is None:
            if runTime < 500:
                message = "Default runtime not chosen, therefore default shocks cannot be chosen"
                logging.error(message)
                raise Exception(message)
            shocks = np.array([[50, 1], [250, -.8]])
        self.shocks = shocks
        logging.info(f"Set shocks to {self.shocks}")
        if not isinstance(shocks, np.ndarray):
            message = "Shocks must be an NumpPy array"
            logging.error(message)
            raise Exception(message)
        for time, value in self.shocks:
            if time > runTime:
                message = "Cannot have shock value after simulation ends"
                logging.error(message)
                raise Exception(message)
            if time < 0:
                message = "Cannot have shock value in time period before 0"
                logging.error(message)
                raise Exception(message)
        rows, cols = shocks.shape
        if rows > runTime:
            message = "More shock periods than total time periods"
            logging.error(message)
            raise Exception(message)

        self.smoothingValue = smoothingValue
        logging.info(f"Set smoothingValue to {self.smoothingValue}")
        if not isinstance(smoothingValue, int):
            message = "Smoothing value must be an int"
            logging.error(message)
            raise TypeError(message)
        if smoothingValue < 1:
            message = "Smoothing value cannot be less than 1"
            logging.error(message)
            raise Exception(message)

        self.wholesalerProfileOption = wholesalerProfileOption  # Options are optimization or default (no optimization)
        logging.info(f"Set wholesalerProfileOption to {self.wholesalerProfileOption}")
        if wholesalerProfile is None:
            wholesalerProfile = [0.5, 0.5]
        self.wholesalerProfile = wholesalerProfile  # Allocation of orders for wholesaler
        logging.info(f"Set wholesalerProfile to {self.wholesalerProfile}")
        if len(self.wholesalerProfile) != 2:
            message = "Length of wholesaler profile must be 2"
            logging.error(message)
            raise Exception(message)
        if sum(self.wholesalerProfile) != 1:
            message = "Sum of wholesaler profile must equal 1"
            logging.error(message)
            raise Exception(message)
        
        self.cost_per_acre = 5000            # USD per acre of lettuce farming
        self.cpm_us = 2.73                    # USD cost per mile in the US
        self.cpm_mexico = 2.36                # USD cost per mile in Mexico
        self.emission_per_acre = 30         # lbs CO2 per acre. (right now this number is made up)
        self.emission_per_mile = 3.73333       # lbs CO2 per mile
        self.water_per_acre = 792,000          # gallons per acre annually. (this needs to be adjusted to account per truck somehow)
        logging.info("Set fixed cost/environment variables: cost_per_acre=5000, cpm_us=0.6, cpm_mexico=0.4, emission_per_acre=30, emission_per_mile=0.12, water_per_acre=1500")
    # def resetDefault(self):
    #     """
    #     Resets the configurations to their default values
    #
    #     :return:
    #     """
    #     self.__init__()

    # def resetToValues(self, config):
    #     """
    #     Resets the configurations to the specified values. Used when not wanting to reset to default.
    #
    #     :param Configs config:
    #         A config object that has the values wanting to reset to
    #
    #     :return:
    #     """
    #     self.__init__(runTime=config.runTime,
    #                   historyTime=config.historyTime,
    #                   shipDelayOption=config.shipDelayOption,
    #                   shipDelayValue=config.shipDelayValue,
    #                   shipDelayList=config.shipDelayList,
    #                   demandMu=config.demandMu,
    #                   demandStd=config.demandStd,
    #                   wholesalerProfileOption=config.wholesalerProfileOption,
    #                   wholesalerProfile=config.wholesalerProfile,
    #                   shocks=config.shocks,
    #                   alphaOption=config.alphaOption,
    #                   alphaValue=config.alphaValue,
    #                   alphaValueList=config.alphaValueList,
    #                   smoothingValue=config.smoothingValue,
    #                   numTimePeriodsForCalc=config.numTimePeriodsForCalc,
    #                   covarianceMatrix=config.covarianceMatrix,
    #                   riskTolerance=config.riskTolerance,
    #                   historicalReturnRuleOption=config.historicalReturnRuleOption
    #                   )

    # def experimentName(self):
    #    return f"{self.}"

    def __getitem__(self, item):
        return self.__dict__[item]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)