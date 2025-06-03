import itertools
import logging
import matplotlib.pyplot as plt
import time
import math 
import numpy as np
from numpy import random as rand

from PO import PO
import Firm
#from Configs import Configs


# https://pynative.com/python-class-variables/#:~:text=In%20Python%2C%20Class%20variables%20are,all%20objects%20of%20the%20class.
# Consider changing some variables to class variables shared by all instances of the class
# Candidates: testOption(not currently used), numAgents, runTime, historyTime, agentList, shipDelay, shocks, totalTime
# demandMu, demandStd, smoothingValue, wholesalerProfile
# This is likely the right thing to do as each new simulation object is typically called as a new iteration, where all
# parameters are exactly the same and we average the results after the iterations are done
# Also consider calling config and within config establishing the simulation class/changing simulation values
class Simulation:
    """
    A class to run an agent-based simulation.


    Class Attributes
    ---------
    idIter: iterator
        Creates a new iterator that will be used to create a unique Identifier for the simulation.
        Will increment each time a new object is created.
    shipDelay: int
        The number of time periods it takes for product to be received. Ordered product is received in the time period
        (t + shipDelayValue). An important implication is that there will be (shipDelayValue - 1) time periods of sales
        while the product is en route.
        Example: (t=10, shipDelayValue=2)
            [t=10] Product ordered at end of time 10
            [t=11] sales actualized for time 11
            [t=12] Product received at beginning of time 12
    demandMu: int
        Value representing the average end consumer demand (i.e. the demand signal to firm 0). Every time period a
        random draw from a normal distribution is made with mu=demandMu and standard deviation=demandStd. In other words,
        end consumer demand is a random variable ~N(demandMu, demandStd)
    demandStd: int
        Value representing the standard deviation in end consumer demand (i.e. the demand signal to firm 0). Every time period a
        random draw from a normal distribution is made with mu=demandMu and standard deviation=demandStd. In other words,
        end consumer demand is a random variable ~N(demandMu, demandStd)
    shocks: NumPy 2D Array
        An array containing the time periods (column 1) and shock value (column 2; as a percentage change)

        Example: [[10, 0.2][20, -0.3]] Has a shock of 20% increase in time period 10 and a shock of 30% down in time
        period 20. If the demand value for both time periods was originally 10, the new value would be 1.2 and 0.7
        respectively
    wholesalerProfile: list
        A list where each entry is a float representing the percentage of supplies that will be ordered from a
        particular supplier. Must sum to 1.

        Currently the simulation only allows two suppliers for the wholesaler. Thus, index 0
        is the percentage firm 1 (wholesaler) buys from firm 2, and index 1 is the percentage firm 1 buys from firm
        3.

        In the special case where wholesalerProfileOption = 'Optimized', this profile changes
        every time period. Otherwise, it stays constant the entire simulation.
    smoothingValue: int
        smoothingValue takes a value of 1 or 2.
        Choosing 1 turns off smoothing (future demand = current time period demand).
        Choosing 2 turns on smoothing of 1 time period
            [future demand = (current demand + current forecasted demand) / 2]
    numAgents: int
        Parameter value for the number of agents in the initialized simulation
    runTime: int
        The amount of time periods the simulation will run (not including historical time).
    historyTime: int
        Number of historical time periods to create before simulation runs. Used for some calculations.
    totalTime: int
        The total time periods including history time. totalTime = historyTime + runTime
    testOption: Bool
        (Currently not implemented) Sets the simulation into test mode, enabling some features and disabling others

    Class Methods
    ---------
    initializeClass(configs, testOption)
        Sets class-level attributes to the values specified in the configs object.

    Instance Attributes
    ---------
    id: int
        Unique identifier for the simulation
    rng: rng
        Establishes for each simulation a new rng generator object
    timePeriod: int
        The current time period for the simulation. Primarily used for data arrays that do not have historical data.
    timeIndex: int
        The current value used for indices in various data arrays that also have historical data.
        timeIndex = timePeriod + historyTime
    agentList: list
        A list of the agent objects
    activePoList: list
        A list of PO objects considered active. Closed PO's move to the closedPoList. Keeping this list short
        improves processing time.
    closedPoList: list
        A list of PO objects considered closed.
    demandArray np array:
        An array used for internal simulation processing. Contains the end consumer demand values (i.e. the demand
        for firm 0) each time  period. Established at simulation initialization and never changed thereafter.
    dataSet np array:
        An m x t x 18 array, where m = number of agents, t = total runtime (not including historical time), and 18
        is the number of individual data (e.g. beginning_wip_inventory, total_wip_ordered, etc.) See Firm3
        documentation in __init__ for ledger columns.
    demandData np array:
        An m x t array, where m = number of agents and t = total runtime (not including historical time). Contains
        longitudinal data where each entry is a firm-timeperiod demand value.

    Instance Methods
    ---------
    initializeSim(testOption):
        Sets up the simulation in time period 0. Calls methods intitializeAgents() and createHistory()
    initializeAgents():
        Initializes the agents for time period 0, setting up historical demand values. Calls the Firm3 function
        initializeAgent()
    createHistory():
        Creates data for historical timeperiods. Historical time periods are referred to in data arrays with negative
        numbers. Calls methods createDemandArray(), createHistoricalPos(), assignHistoricalPos()
    createDemandArray():
        Will create a 1D array of demand values. First n elements are historical demand and next m elements are
        future demand. Also inputs the shock values
    createHistoricalPos():
        A method to instantiate PO objects for each firm for each time period. Wholesaler and Manufacturer firms have
        some uniqueness that necessitates completing the process differently.
    assignHistoricalPos():
        A method that sends an individual PO to the supplier and customer listed on the PO. There are several firm-level
        methods that require this object to also  be available at the firm-level.
    inputShocks():
        Shocks are a percent deviation from the demand average. They are exogenous and set by the user.
    runSimulation():
        The main engine for the simulation. Loops through t times, where t= runTime (not including historical time). Also
        keeps track of simulation time performance by primary function called (see list below 1-7 for descriptions of the
        'primary functions')

        In each time period, the following actions by a firm are performed  (in the following order):
                1. receive orders for supplies that were sent to suppliers in a previous time period
                2. Realize current time period customer demand
                3. Produce finished goods from work in progress inventory based on customer orders
                4. Send finished goods to customers
                5. Update future demand forecasts, used when  ordering supplies
                6. Place new orders  for supplies
                7. End of time period housekeeping (for simulation and internal firm purposes)
    createDemandPo():
        Creates a PO object that will be sent to the retailer representing the current time periods end consumer demand
    receiveShipments(agent):
        A function to represent the real-life process of receiving supply shipments

        The simulation will loop through the active PO list and match the focal agent with the PO's where they are
        listed as customers. It then calls the Firm3 function receiveWipOrder(...) which handles internal processing
        of the WIP inventory.
    actualizeDemand(agent):
        A  function to represent the real-life process of receiving customer orders

        The simulation will loop through the active PO list and match the focal agent with the PO's where they are
        listed as suppliers. It also checks to ensure that the time period the order was placed matches the current
        time period. This uses the assumption that orders placed are received by the supplier in the same time period.
        The function gathers up all such PO's (for some firms there will be multiple orders). It then calls the Firm3
        function receiveCustomerDemand(...) which handles internal processing of the order.
    production(agent):
        A  function to represent the real-life production process

        The simulation calls the Firm3 function production(...) which handles the internal production process.
    sendShipments(agent):
        A  function to represent the real-life process of shipping finished goods to customers

        The simulation calls the Firm3 function sendCustomerShipments(...) which handles the internal shipping process.
    updateForecast(agent):
        A  function to represent the real-life process of updating future demand forecasts

        The simulation calls the Firm3 function updateDemandForecast(...) which handles the internal forecasting process.
        The class variable smoothingValue is sent as an option variable.  smoothingValue takes a value of 1 or 2.
        Choosing 1 turns off smoothing (future demand = current time period demand).
        Choosing 2 turns on smoothing of 1 time period [future demand = (current demand + current forecasted demand) / 2]
    supplyOrders(agent)
        A  function to represent the real-life process of generating supply orders

        The simulation will call the Firm3 function orderSupplies(...) which handles the internal supply ordering
        process. orderSupplies(...) returns a list of PO objects. Then all POs in the received list are added to the
        simulations list of active POs.
    endOfDay():
        A  function used in the simulation to gather current time period information and then update internal logs.

        The simulation will call the Firm3 function endOfDay(...) which handles the internal end of day process.
        Then the active po list is looped through and closes the current time period's PO that was created to represent
        the end consumer, with the retailer as the supplier.
    cleanActivePoList():
        Removes all closed POs from the active PO list and puts it on closed list. This is intended to reduce time
        looping through the active PO list each time period to find a specific PO.
    advanceTime():
        Performs end of time period functions and advances the time period
    resetFirms():
        Resets the firms for a new simulation.
    processData(simNo):
        Calls getData() function and processes for analysis.
    getData():
        Calls firm method sendData and saves data to simulation-level array
    plotData(simNo):
        Plots data for visualization of results
    sample_normal(m, s):
        Samples from the normal distribution with a mean of m and standard deviation of s
    """
    #idIter = itertools.count()
    shipDelay = None
    demandMu = None
    demandStd = None
    shocks = None
    wholesalerProfile = None
    smoothingValue = None
    numAgents = None
    runTime = None
    historyTime = None
    totalTime = None
    testOption = False
    agentList = None

    @classmethod
    def initializeClass(cls, configs, agentList, testOption=False):
        cls.shipDelay = configs.shipDelayList
        cls.demandMu = configs.demandMu
        cls.demandStd = configs.demandStd
        cls.shocks = configs.shocks
        cls.wholesalerProfile = configs.wholesalerProfile
        cls.smoothingValue = configs.smoothingValue
        cls.numAgents = configs.numAgents
        cls.testOption = testOption
        cls.runTime = configs.runTime
        cls.historyTime = configs.historyTime
        cls.totalTime = cls.runTime + cls.historyTime
        cls.agentList = agentList
        cls.miles_mexico   = configs.miles_mexico
        cls.miles_us       = configs.miles_us
        cls.cpm_mexico     = configs.cpm_mexico
        cls.cpm_us         = configs.cpm_us


    def __init__(self):
        #self._id = next(self.idIter)  # Creates a new unique identifier for the simulation
        self._rng = rand.default_rng()
        self._timePeriod = 0  # Every simulation starts at time period 0
        if self.historyTime is None:
            raise Exception("Class not initialized")
        self._timeIndex = self._timePeriod + self.historyTime
        # Because there are historical time periods identified with negative numbers, the time period will not match up
        # with the index value for data. For example, in a sim with 2 historical time periods, time period 0 (beginning
        # of actual simulation dynamics) would be index 2 in data arrays

        self._activePoList = []
        self._closedPoList = []
        self._demandArray = np.zeros(self.totalTime, dtype=int)
        self._dataSet = []
        self._demandData = []

        self._truckCount = np.zeros(self.totalTime,dtype = int) #The amount of fuel used per mile is also per truck
        self._transportCost = np.zeros(self.totalTime, dtype = float)

    def initializeSim(self, testOption=False):
        """
        Initializes the simulation for time period 0

        :param bool testOption:
            If True, will run the simulation in test mode
        :return:
        """
        tic = time.perf_counter()
        if testOption:
            Simulation.testOption = True
            logging.warning("Simulation is in test mode")
        self.initializeAgents()
        self.createHistory()
        toc = time.perf_counter()
        return toc - tic

    def initializeAgents(self):
        """
        Initializes the agents in time period 0. Manufacturer needs a different initialization, and therefore is called
        differently.

        :return:
        """
        logging.info('Initializing %s agents...', self.numAgents)
        for agent in self.agentList:
            if isinstance(agent, Firm.Manufacturer):
                historicalWeeklyDemand = agent.calculateHistoricalDemand(self.demandMu, self.wholesalerProfile)
            else:
                historicalWeeklyDemand = agent.calculateHistoricalDemand(self.demandMu)
            agent.initializeAgent(historicalWeeklyDemand)

    def createHistory(self):
        """
        Creates data for historical time periods. Historical time periods are referred to in data arrays with negative
        numbers.

        :return:
        """
        self.createDemandArray()
        self.createHistoricalPos()
        self.assignHistoricalPos()

    def createHistoricalPos(self):
        """
        A method to instantiate PO objects for each firm for each time period. Wholesaler and Manufacturer firms have
        some uniqueness that necessitates completing the process differently.

        :return:
        """
        # Create Historical Active PO's
        for agent in self.agentList:
            if agent.supplierList is None:
                raise Exception("No suppliers initialized for firm %s", agent.id)
            elif isinstance(agent, Firm.Wholesaler):
                for timePeriod in range(agent.shipDelay):
                    shipDate = -(timePeriod + 1)
                    orderList = agent.calculateWipOrder(self.demandMu)
                    for index, order in enumerate(orderList):
                        newPo = PO(agent.id, agent.supplierList[index], order, shipDate)
                        newPo.updatePO({'arrivalTime': shipDate + agent.shipDelay,
                                        'fulfilledTime': shipDate,
                                        'fulfilledAmt': order})
                        newPo.supplierClosed = True
                        self._activePoList.append(newPo)
                        logging.debug("Created a historical PO for firm %s to arrive in time period %s",
                                     agent.id, newPo.arrivalTime)
            else:
                for timePeriod in range(agent.shipDelay):
                    if isinstance(agent, Firm.Manufacturer):
                        orderAmt = agent.calculateHistoricalDemand(self.demandMu,
                                                                   wholesalerProfile=self.wholesalerProfile)
                    else:
                        orderAmt = max(0, int(self.demandMu / len(agent.supplierList)))
                    for supplier in agent.supplierList:
                        shipDate = -(timePeriod + 1)
                        newPo = PO(agent.id, supplier, orderAmt, shipDate)
                        newPo.updatePO({'arrivalTime': shipDate + agent.shipDelay,
                                        'fulfilledTime': shipDate,
                                        'fulfilledAmt': orderAmt})
                        newPo.supplierClosed = True
                        self._activePoList.append(newPo)
                        logging.debug("Created a historical PO for firm %s to arrive in time period %s",
                                     agent.id, newPo.arrivalTime)

    def assignHistoricalPos(self):
        """
        A method that sends an individual PO to the supplier and customer listed on the PO. There are several firm-level
        methods that require this object to also  be available at the firm-level.

        :return:
        """
        for agent in self.agentList:
            custPos = []
            supPos = []
            for po in self._activePoList:
                if po.supplier == agent.id:
                    custPos.append(po)
                elif po.customer == agent.id:
                    supPos.append(po)
            agent.createHistory(custPos, supPos)

    def createDemandArray(self):
        """
        Will create a 1D array of demand values. First n elements are historical demand and next m elements are
        future demand. Also inputs the shock values

        :return:
        """
        # Used for logging
        count = 0
        low = self.demandMu
        high = self.demandMu
        # Create the historical demand
        for i in range(self.historyTime):
            self._demandArray[i] = self.demandMu
            count += 1
        # Create the future demand. Starts where historical demand ended
        for i in range(self.historyTime, self.totalTime):
            demand = max(0, int(self.sample_normal(self.demandMu, self.demandStd)))
            if demand < low:
                low = demand
            elif demand > high:
                high = demand
            self._demandArray[i] = demand
            count += 1
        # Error Checking and logging
        if count != self.totalTime:
            logging.warning("Issue creating demand array. Total time does not match times demand was changed")
        logging.debug(
            "Created demand array with %s total time periods. Low value: %s, High value: %s (not including "
            "shocks).", self.totalTime, low, high)
        self.inputShocks()

    def inputShocks(self):
        """
        Shocks are a percent deviation from the demand average. They are exogenous and set by the user.

        :return:
        """
        # Only enters if the shock array was defined
        if self.shocks is not None:  # Input the shocks
            countShock = 0
            for i, t in enumerate(self.shocks[:, 0]):
                timePeriod = int(t + self.historyTime)  # Have to account for history time
                shockValue = int(self.demandMu * self.shocks[i, 1])
                self._demandArray[timePeriod] = self.demandMu + shockValue
                countShock += 1
            logging.debug("Updated %s time periods with shocks.", countShock)

    def runSimulation(self):
        """
        The main engine for the simulation. Loops through t times, where t= runTime (not including historical time). Also
        keeps track of simulation time performance by primary function called (see list below 1-7 for descriptions of the
        'primary functions')

        In each time period, the following actions by a firm are performed  (in the following order):
                1. receive orders for supplies that were sent to suppliers in a previous time period
                2. Realize current time period customer demand
                3. Produce finished goods from work in progress inventory based on customer orders
                4. Send finished goods to customers
                5. Update future demand forecasts, used when  ordering supplies
                6. Place new orders  for supplies
                7. End of time period housekeeping (for simulation and internal firm purposes)
        :return:
        """
        logging.info("Running the simulation...")
        timeList = np.zeros(10)
        while True:
            timeList[0] += self.createDemandPO()  # Looks at demand array and creates a demand PO
            for agent in self.agentList:
                logging.debug('Beginning day %s for firm %s', self._timePeriod, agent.id)
                timeList[1] += self.beginningOfDay(agent)
                timeList[2] += self.receiveShipments(agent)
                timeList[3] += self.actualizeDemand(agent)
                timeList[4] += self.production(agent)
                timeList[5] += self.sendShipments(agent)
                timeList[6] += self.updateForecast(agent)
                timeList[7] += self.supplyOrders(agent)
                timeList[8] += self.endOfDay(agent)
                logging.debug('Ending day %s for firm %s', self._timePeriod, agent.id)
            timeList[9] += self.cleanActivePoList()
            condition = self.advanceTime()  # Returns false if the final time period
            if not condition:  # Ends the while loop if final time period
                logging.info("Ending simulation...")
                break
        return timeList

    def createDemandPO(self):
        """
        Creates a PO object that will be sent to the retailer representing the current time periods end consumer demand

        :return:
            A float timing the performance of the function
        """
        # Baseline  performance for 1 sim: 0.16s (optimized) 0.14s (default)
        # Current  performance for 1 sim: 0.13s (optimized) 0.13s (default)
        tic = time.perf_counter()
        for i, demand in enumerate(self._demandArray):
            if i == self._timeIndex:  # Demand array starts at historical start
                newPo = PO(-99, 0, demand, self._timePeriod)
                newPo.updatePO({'arrivalTime': self._timePeriod})
                self._activePoList.append(newPo)

                trucks = int(math.ceil(demand / 38))
                cost_per_truck_money = (self.miles_mexico * self.cpm_mexico) + (self.miles_us * self.cpm_us)

                total_moneyCost = trucks * cost_per_truck_money

                yield_per_acre = 30,000 #An acre of farmland yields 30k lbs of butter lettuce annually.
                acres_per_truck = 38,000/30,000 #A truck can holod 38k lbs of butter lettuce. (it runs out of space before weight carrying capacity)

                #CO2 cost
                co2_from_acres = acres_per_truck * 13889.13 #13889.13 is the pounds of CO2 produced by an acre of butter lettuce annually
                co2_from_travel = (Simulation.miles_mexico + Simulation.miles_us)*6*22.4 #22.4 lbs c02/mile, 6mpg

                total_co2Cost = trucks*(co2_from_acres+co2_from_travel)

                #Water cost
                water_per_acre = 32,000 
                water_per_truck = water_per_acre * acres_per_truck
                total_waterCost = water_per_truck*trucks



                self._truckCount[self._timeIndex] = trucks
                self._transportCost[self._timeIndex] = total_moneyCost 

                retailer = self.agentList[0]  # by construction, firm 0 is always the retailer
                retailer.actualizeCost(
                    total_moneyCost,
                    total_co2Cost,
                    total_waterCost
                )

                logging.debug("Created a demand PO for current time period %s", self._timePeriod)
                break  # Only one demand PO created for each time period currently (Changes with multiple retailers)
        toc = time.perf_counter()
        return toc - tic

    @staticmethod
    def beginningOfDay(agent):
        tic = time.perf_counter()
        agent.beginningOfDay()
        toc = time.perf_counter()
        return toc - tic

    def receiveShipments(self, agent):
        """
        A function to represent the real-life process of receiving supply shipments

        The simulation will loop through the active PO list and match the focal agent with the PO's where they are
        listed as customers. It then calls the Firm3 function receiveWipOrder(...) which handles internal processing
        of the WIP inventory.

        :param Firm3.Firm3 agent:
            A firm object
        :return:
            A  float timing the performance of the function
        """
        # Baseline  performance for 1 sim: 0.21s (optimized) 0.11s (default)
        # Current  performance for 1 sim: 0.17s (optimized) 0.10s (default)
        tic = time.perf_counter()
        warnCust = True
        warnTime = True
        # Loop through active PO list to see if agent will receive shipment in current time period
        for po in self._activePoList:
            # Check the customer id on the PO and match with the current agent id
            if po.customer == agent.id and po.arrivalTime == self._timePeriod:
                agent.receiveWipOrder(po.id)
                warnTime = False
                warnCust = False
        # Log potential problems
        if warnCust:
            logging.warning('No outstanding WIP shipments for firm %s in active PO list', agent.id)
        if warnTime:
            logging.warning('No incoming WIP shipments for firm %s in time %s', agent.id, self._timePeriod)
        toc = time.perf_counter()
        return toc - tic

    def actualizeDemand(self, agent):
        """
        A  function to represent the real-life process of receiving customer orders

        The simulation will loop through the active PO list and match the focal agent with the PO's where they are
        listed as suppliers. It also checks to ensure that the time period the order was placed matches the current
        time period. This uses the assumption that orders placed are received by the supplier in the same time period.
        The function gathers up all such PO's (for some firms there will be multiple orders). It then calls the Firm3
        function receiveCustomerDemand(...) which handles internal processing of the order.

        :param Firm3.Firm3 agent:
            A firm object
        :return:
            A  float timing the performance of the function
        """
        # Baseline  performance for 1 sim: 0.30s (optimized) 0.26s (default)
        # Current  performance for 1 sim: 0.25s (optimized) 0.25s (default)
        tic = time.perf_counter()
        # Loop through active PO list to see if agent has received a customer order in current time period
        totPos = []
        for po in self._activePoList:
            if po.supplier == agent.id and po.orderTime == self._timePeriod:
                totPos.append(po)
        agent.receiveCustomerDemand(totPos)  # Send all the POs
        toc = time.perf_counter()
        return toc - tic

    @staticmethod
    def production(agent):
        """
        A  function to represent the real-life production process

        The simulation calls the Firm3 function production(...) which handles the internal production process.

        :param Firm3.Firm3 agent:
            A firm object
        :return:
            A  float timing the performance of the function
        """
        # Baseline  performance for 1 sim: 0.43s (optimized) 0.43s (default)
        # Current  performance for 1 sim: 0.34s (optimized) 0.41s (default)
        tic = time.perf_counter()
        agent.production()
        toc = time.perf_counter()
        return toc - tic

    def sendShipments(self, agent):
        """
        A  function to represent the real-life process of shipping finished goods to customers

        The simulation calls the Firm3 function sendCustomerShipments(...) which handles the internal shipping process.

        :param Firm3.Firm3 agent:
            A firm object
        :return:
            A  float timing the performance of the function
        """
        # Baseline  performance for 1 sim: 0.25s (optimized) 0.22s (default)
        # Current  performance for 1 sim: 0.18s (optimized) 0.21s (default)
        tic = time.perf_counter()
        agent.sendCustomerShipments(self._timePeriod)
        toc = time.perf_counter()
        return toc - tic  # Current  performance for 1 sim: 0.25s (optimized) 0.22s (default)

    def updateForecast(self, agent):
        """
        A  function to represent the real-life process of updating future demand forecasts

        The simulation calls the Firm3 function updateDemandForecast(...) which handles the internal forecasting process.
        The class variable smoothingValue is sent as an option variable.  smoothingValue takes a value of 1 or 2.
        Choosing 1 turns off smoothing (future demand = current time period demand).
        Choosing 2 turns on smoothing of 1 time period [future demand = (current demand + current forecasted demand) / 2]

        :param Firm3.Firm3 agent:
            A firm object
        :return:
            A  float timing the performance of the function
        """
        # Baseline  performance for 1 sim: 0.43s (optimized) 0.37s (default)
        # Current  performance for 1 sim: 0.18s (optimized) 0.17s (default)
        tic = time.perf_counter()
        agent.updateDemandForecast(self.smoothingValue)
        toc = time.perf_counter()
        return toc - tic

    def supplyOrders(self, agent):
        """
        A  function to represent the real-life process of generating supply orders

        The simulation will call the Firm3 function orderSupplies(...) which handles the internal supply ordering
        process. orderSupplies(...) returns a list of PO objects. Then all POs in the received list are added to the
        simulations list of active POs.

        :param Firm3.Firm3 agent:
            A firm object
        :return:
            A  float timing the performance of the function
        """
        # This function is very time intensive for the optimized condition
        tic = time.perf_counter()
        poList = agent.orderSupplies(self._timePeriod)
        for po in poList:
            self._activePoList.append(po)
        toc = time.perf_counter()
        return toc - tic  # Current  performance for 1 sim: 5.52s (optimized) 0.39s (default)

    def endOfDay(self, agent):
        """
        A  function used in the simulation to gather current time period information and then update internal logs.

        The simulation will call the Firm3 function endOfDay(...) which handles the internal end of day process.
        Then the active po list is looped through and closes the current time period's PO that was created to represent
        the end consumer, with the retailer as the supplier.

        :param Firm3.Firm3 agent:
            A firm object
        :return:
            A  float timing the performance of the function
        """
        tic = time.perf_counter()
        agent.endOfDay()
        for po in self._activePoList:  # Close the demand POs
            if po.customer == -99 and po.orderTime == self._timePeriod:
                po.customerClosed = True
        toc = time.perf_counter()
        return toc - tic  # Current  performance for 1 sim: 0.12s (optimized) 0.11s (default)

    def cleanActivePoList(self):
        """
        Removes all closed POs from the active PO list and puts it on closed list. This is intended to reduce time
        looping through the active PO list each time period to find a specific PO.

        :return:
            A  float timing the performance of the function
        """
        tic = time.perf_counter()
        # First add the closed PO's to closed PO list
        count1 = 0
        begLen = len(self._activePoList)
        for po in self._activePoList:
            if po.closePO(self._timePeriod):
                self._closedPoList.append(po)
                count1 += 1
        # Then remake the active list to not include the closed PO's
        self._activePoList[:] = [po for po in self._activePoList if not po.closed]
        if count1 > 0:
            logging.debug('Removed %s closed POs from active list. Active list now is of size %s (was %s)', count1,
                         len(self._activePoList), begLen)
        else:
            logging.warning('No POs were removed from active list.')
        toc = time.perf_counter()
        return toc - tic  # Current  performance for 1 sim is effectively 0s

    def advanceTime(self):
        """
        Performs end of time period functions and advances the time period

        :return: False if current time period is final time period, true otherwise
        """
        logging.debug('Advancing time from %s to %s', self._timePeriod, self._timePeriod + 1)
        self._timePeriod += 1
        self._timeIndex += 1
        if self._timePeriod == self.runTime:
            return False
        else:
            return True

    def resetFirms(self):
        """
        Resets the firms for a new simulation.

        :return:
        """
        for agent in self.agentList:
            agent.resetFirm(self.demandMu)

    def processData(self, simNo):
        """
        Calls getData() function and processes for analysis.
        :param simNo:
        :return:
        """
        self.getData()
        # self.plotData(simNo)
        return self._demandData

    def getData(self):
        """
        Calls firm method sendData and saves data to simulation-level array
        :return:
        """
        demandDataList = []
        for agent in self.agentList:
            demandData = agent.sendData('Demand')
            demandDataList.append(demandData)
        self._demandData = np.array(demandDataList)

        costMoneyList = []
        for agent in self.agentList:
            costMoneyData = agent.sendData('CostMoney')
            costMoneyList.append(costMoneyData)
        self._costMoneyData = np.array(costMoneyList)

        # 3) Build and store costCO2Data = (numAgents × totalTime)
        costCO2List = []
        for agent in self.agentList:
            costCO2Data = agent.sendData('CostCO2')
            costCO2List.append(costCO2Data)
        self._costCO2Data = np.array(costCO2List)

        # 4) Build and store costWaterData = (numAgents × totalTime)
        costWaterList = []
        for agent in self.agentList:
            costWaterData = agent.sendData('CostWater')
            costWaterList.append(costWaterData)
        self._costWaterData = np.array(costWaterList)

        return self._demandData, self._costMoneyData, self._costCO2Data, self._costWaterData

    def plotData(self, simNo):
        """
        Plots data for visualization of results
        :param simNo:
        :return:
        """
        # Demand Data
        plt.figure(1)
        chartTitle = f"Demand Chart, Sim {simNo}"
        plt.title(chartTitle)
        plt.xlabel('Time')
        plt.ylabel('Demand')
        xAxis = self._dataSet[0, self.historyTime:, 0]  # Time axis
        for firm in range(len(self._dataSet)):
            data = self._dataSet[firm, self.historyTime::, 17]
            label = f"Firm {firm}"
            plt.plot(xAxis, data, label=label)
        plt.legend()
        plt.show()

        # # Inventory Data
        # plt.figure(2)
        # plt.title('WIP Inventory')
        # plt.xlabel('Time')
        # plt.ylabel('Units')
        # xAxis = self._dataSet[0, self._historyTime:, 0]  # Time axis
        # for firm in range(len(self._dataSet)):
        #     data = self._dataSet[firm, self._historyTime::, 1]
        #     label = f"Firm {firm}"
        #     plt.plot(xAxis, data, label=label)
        # plt.legend()
        # plt.show()

    def sample_normal(self, m, s):
        """
        Samples from the normal distribution with a mean of m and standard deviation of s

        :param int m:
            Average
        :param int s:
            standard deviation
        :return:
        """
        return self._rng.normal(m, s)