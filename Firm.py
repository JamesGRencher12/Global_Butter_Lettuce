import itertools
import numpy as np
import logging
import pandas as pd
import random
from scipy.optimize import minimize, Bounds

import PO


class Firm:
    """
    A class to represent a firm.

    Class Attributes
    ---------
    idIter: iterator
        Creates a new iterator that will be used to create a unique Identifier for the firm.
        Will increment each time a new object is created.

    Instance Attributes
    ---------
    actualizedDemandArray: np.ndarray
        An array of actualized demand
    alpha: float
        Value representing the percent of WIP inventory maintained as safety stock.
    closedCustomerPos: list
        A list of all the customer PO's that have been closed (by both supplier and customer). A customer PO is one that
        originated from the focal firm's customers.
    closedSupplierPos: list
        A list of all the supplier PO's that have been closed (by both supplier and customer). A supplier PO is one that
        originated from this firm and sent to its supplier.
    customerList: list
        A list of integers signifying the firm number for each of their customers. End consumers = '-99'
    customerPolist: list
        Running list of all PO objects sent to this firm as customer orders
    demandForecastArray: np.ndarray
        An array of demand forecasts. Several rules and calculations rely on these forecasts.
    desiredWipInventory: int
        A floor amount signifying the minimum WIP inventory the firm wants to maintain. A function of alpha. Updates
        each time period based on demand forecasts.
    fgInventory: int
        Running inventory of finished goods (i.e. goods ready to be sent to customers).
    historyTime: int
        Number of historical time periods to create before simulation runs. Used for some calculations.
    id: int
        Unique identifier for the firm
    ledger: np.ndarray
        A ledger of information used for internal firm purposes, including calculations, optimization, etc. Also used
        for simulation error checking.
        Ledger Columns:
                [0] time
                [1] beginning_wip_inventory
                [2] total_wip_received
                [3] production_wip_inventory (Inventory at beginning of production)
                [4] wip_used_in_production
                [5] ending_wip_inventory
                [6] total_wip_ordered
                [7] beginning_fg_inventory
                [8] ???????
                [9] total_new_production_orders
                [10] total_fg_produced
                [11] fg_inventory_after_production
                [12] total_fg_shipped
                [13] ending_fg_inventory
                [14] wip_in_transit
                [15] desired_wip_inventory
                [16] forecasted_demand
                [17] actual_demand
    poList: list
        Running list of all PO objects created by this firm to order supplys
    producedPos: list
        A list of PO objects that have been produced.
    productionOrder: int
        A number representing the total finished goods requested to be produced in the current time period.
    productionQueue: list
        A list of PO objects that are waiting to be fulfilled.
    runTime: int
        The amount of time periods the simulation will run (not including historical time)
    shipDelay: int
        The number of time periods it takes for product to be received. Ordered product is received in the time period
        t + shipDelayValue. An important implication is that there will be (shipDelayValue - 1) time periods of sales
        while the product is en route.
        Example: (t=10, shipDelayValue=2)
            [t=10] Product ordered at end of time 10
            [t=11] sales actualized for time 11
            [t=12] Product received at beginning of time 12
    shippedPos: list
        A list of PO objects that have been produced and also shipped.
    shippingQueue: list
        A list of PO objects that have been produced and are waiting to be shipped.
    supplierList: list
        A list of integers signifying the firm number for each of the focal firm's suppliers. Raw materials supplier
        is '-99'
    timeIndex: int
        The current value used for indices in various data arrays that also have historical data.
        timeIndex = timePeriod + historyTime
    timePeriod: int
        The current time period for the simulation (not including historyTime). Primarily used for data arrays that do
        not have historical data.
    totalTime: int
        The total time periods including history time. totalTime = historyTime + runTime
    wipInventory: int
        Running inventory of work in process (WIP) materials (synonymous in this sim for raw materials)


    Instance Methods
    -------
    actualizeDemand(amount):
        Records the time period's demand information.
    beginningOfDay():
        Basic function, whose primary purpose is to be overridden by the wholesaler class
    calculateFutureDemand(smoothingValue):
        A function to calculate future demand predictions. This function incorporates demand smoothing, which has been
        shown in the literature to reduce variability in orders. Demand smoothing is implemented particularly by taking
        the average of the current time period's forecasted and the current time period's actual demand.
    calculateHistoricalDemand(demandMu):
        Basic function, whose primary purpose is to be overridden by the Manufacturer class
    calculateSalesDuringDelay():
        A function to calculate the amount of sales that will occur during the time that one WIP order is being shipped
    calculateSupplyInTransit():
        A function to calculate the amount of WIP that is still in transit, expected to be received. This value is used
        when making ordering decisions.
    calculateSupplyOrder():
        A function that will calculate the total amount of WIP required, taking into account desired WIP, current WIP,
        how many sales are expected over the shipping delay time, and the WIP already in transit.
    chooseSuppliers(currentTimePeriod):
        A basic function whose primary purpose is to be overridden by the Wholesaler class.
    createHistory(custPos, supPos):
        Takes in two lists of po objects and adds it to internal lists
    createPo(supplier, orderAmt, orderTime)
        Creates a new PO object.
    createProductionOrder():
        A method that counts up the order quantities for each customer PO and then puts that integer value into the
        production queue.
    createSupplyOrders(amount):
        A function that will create PO objects based on the desired supply.
    endOfDay():
        Records information in ledger, cleans out PO lists, advances timePeriod and timeIndex.
    initializeAgent(historicalWeeklyDemand):
        Sets up the parameters, attributes and data structures for use in the simulation
    logSupplyPo(poNumber, timeIndex)
        Basic function, whose primary purpose is to be overridden by the Wholesaler class
    makeProduct(poToMake, amount)
        A function to represent a real-life batch order for production.
        It is assumed that no product is lost to spoilage, waste, or defects. That is to say, every firm is assumed to
        be 100% efficient on the transfer from WIP to FG. It is also assumed that there is a 1:1 relationship between
        WIP and FG (i.e. that every finished good only takes one WIP to make the product).
    orderSupplies(currentTimePeriod)
        A method to represent the real-life process of ordering supplies for expected future demand.
    production():
        A function to represent the real-life production decision process. The function looks at its current WIP inventory
        and determines which and how much finished goods to produce.

        The function will loop through the productionQueue one at a time and determine if there is enough WIP inventory
        to fulfill the order. The simulation allows partial fulfillment of orders, but only for the current time period.
        Whatever is not produced in the current time period for a particular PO is cancelled. In other words, a supplier
        cannot partially fulfill an order in t and then fulfill the rest in t+1
    receiveCustomerDemand(customerPOs, time):
        A method to represent the real-life process of receiving customer orders. Turns customer orders into production
        orders.
    receiveWipOrder(poNumber):
        Intakes the WIP order into inventory, checks to see if the PO has been fulfilled.
    resetFirm():
        Resets the firm to default values, in preparation for a new simulation.
    sendCustomerShipments(timePeriod):
        A method to represent the real-life process of sending customer shipments. Updates the PO with the shipped time,
        calculates arrival time and updates PO accordingly. Empties shipping queue when all done.
    sendData(option):
        Returns firm-level data to be aggregated with firm-level data for other firms. Option takes two values, 'Ledger'
        or 'Demand'. Ledger sends the entire ledger, while demand sends only the demand array.
    setClassAttributes(historicalWeeklyDemand):
        Sets (or resets) the initial class attributes
    setDemandArray(historicalWeeklyDemand):
        A method that will set the historical demand values for the demand array
    setForecast(historicalWeeklyDemand):
        Sets up the preliminary demand forecast
    setHistoricalLedger(historicalWeeklyDemand):
        A method that will establish the historical time period values for the firm's internal ledger
    setHistoricalReturnArray(historicalWeeklyDemand):
        Basic function, whose primary purpose is to be overridden by the Wholesaler class
    timeZeroSetup():
        Sets the initial class variables for beginning of time zero
    updateDemandForecast(smoothingValue):
        A function to represent the real-life process of demand  forecasting for a firm.
        A call to calculateFutureDemand(...) is made and an int representing the single-time-period demand forecast is
        returned. desiredWipInventory (a function of alpha and futureDemand) is updated and then recorded in the ledger.
        forecasted demand is then updated in the demandForecastArray for all times: [t+1, t+shipDelay+1]
    warningLoop(listToCheck, funcName)
        A function used for logging purposes. Will check the length of a list that is about to be looped through.
        Will output a warning in the log if the list is longer than 10. This helps to identify potential causes of
        long processing times.
    """
    #idIter = itertools.count()  # Class variable

    def __init__(self,
                 alpha,
                 runTime,
                 historyTime,
                 shipDelay,
                 idNum,
                 customerList=None,
                 customerPoList=None,
                 fgInventory=0,
                 supplierList=None,
                 poList=None,
                 wipInventory=0
                 ):
        # Initializations that are dependent for later initializations
        self._id = idNum
        logging.info(f"Assigned number {self._id} to {type(self)}")

        self._historyTime = historyTime
        logging.debug(f"Firm {self._id}, t=SETUP: Set alpha to {self._historyTime}")
        if not isinstance(historyTime, int):
            message = "historyTime must be an int"
            logging.error(message)
            raise TypeError(message)
        if historyTime < 0:
            message = "historyTime cannot be less than 0"
            logging.error(message)
            raise Exception(message)
        elif historyTime < 5:
            message = f"Value of {historyTime} is low for historyTime. Recommend greater than 5"
            logging.warning(message)

        self._runTime = runTime
        logging.debug(f"Firm {self._id}, t=SETUP: Set runTime to {self._runTime}")
        if not isinstance(runTime, int):
            message = "Runtime must be an int"
            logging.error(message)
            raise TypeError(message)
        if runTime < 0:
            message = "runTime cannot be less than 0"
            logging.error(message)
            raise Exception(message)
        elif runTime < 100:
            message = f"Value of {runTime} is low for runTime. Recommend greater than 100"
            logging.warning(message)

        self._shipDelay = shipDelay
        logging.debug(f"Firm {self._id}, t=SETUP: Set shipDelay to {self._shipDelay}")
        if not isinstance(shipDelay, int):
            message = "shipDelay must be an int"
            logging.error(message)
            raise TypeError(message)
        if shipDelay < 0:
            message = "shipDelay cannot be less than 0"
            logging.error(message)
            raise Exception(message)
        elif shipDelay == 0:
            message = f"Value of {shipDelay} is low for shipDelay. Recommend greater than 0"
            logging.warning(message)
        elif shipDelay > 5:
            message = f"Value of {shipDelay} is high for shipDelay. Recommend not greater than 5"
            logging.warning(message)

        self._timePeriod = 0
        logging.debug(f"Firm {self._id}, t=SETUP: Set initial timePeriod to {self._timePeriod}")

        self._totalTime = int(runTime + historyTime)
        logging.debug(f"Firm {self._id}, t=SETUP: Set totalTime to {self._totalTime}")

        # Remaining initializations are alphabetized
        self._actualizedDemandArray = np.zeros(self._totalTime, dtype=int)
        logging.debug(f"Firm {self._id}, t=SETUP: Established actualizedDemandArray of type "
                      f"{type(self._actualizedDemandArray)} and size {self._totalTime} ")

        self._alpha = alpha
        logging.debug(f"Firm {self._id}, t=SETUP: Set alpha to {self._alpha}")
        if alpha < 0:
            message = "Alpha value cannot be less than zero"
            logging.error(message)
            raise Exception(message)
        if not (isinstance(alpha, float) or isinstance(alpha, int)):
            message = "Alpha value must be either an int or a float"
            logging.error(message)
            raise TypeError(message)

        self._closedCustomerPos = []
        logging.debug(f"Firm {self._id}, t=SETUP: Created empty list closedCustomerPos")

        self._closedSupplierPos = []
        logging.debug(f"Firm {self._id}, t=SETUP: Created empty list closedSupplierPos")

        if customerList is None:
            customerList = []
            logging.debug(f"Firm {self._id}, t=SETUP: Created empty list customerList")
        self.customerList = customerList

        if customerPoList is None:
            customerPoList = []
            logging.debug(f"Firm {self._id}, t=SETUP: Created empty list customerPoList")
        self.customerPoList = customerPoList

        arraySize = self._totalTime + self._shipDelay
        self._demandForecastArray = np.zeros(arraySize, dtype=int)
        logging.debug(f"Firm {self._id}, t=SETUP: Established demandForecastArray of type "
                      f"{type(self._demandForecastArray)} and size {arraySize}")

        self._desiredWipInventory = 0
        logging.debug(f"Firm {self._id}, t=SETUP: Set initial desiredWipInventory to {self._desiredWipInventory}")

        self._fgInventory = fgInventory
        logging.debug(f"Firm {self._id}, t=SETUP: Set initial fgInventory to {self._fgInventory}")
        if not isinstance(fgInventory, int):
            message = "fgInventory must be an int"
            logging.error(message)
            raise TypeError(message)
        if fgInventory < 0:
            message = "fgInventory value cannot be less than zero"
            logging.error(message)
            raise Exception(message)

        self._ledger = np.zeros((self._historyTime, 18), dtype=int)
        """
            Ledger Columns:
                [0] time
                [1] beginning_wip_inventory
                [2] total_wip_received
                [3] production_wip_inventory (Inventory at beginning of production)
                [4] wip_used_in_production
                [5] ending_wip_inventory
                [6] total_wip_ordered
                [7] beginning_fg_inventory
                [8] ???????
                [9] total_new_production_orders
                [10] total_fg_produced
                [11] fg_inventory_after_production
                [12] total_fg_shipped
                [13] ending_fg_inventory
                [14] wip_in_transit
                [15] desired_wip_inventory
                [16] forecasted_demand
                [17] actual_demand
        """
        logging.debug(f"Firm {self._id}, t=SETUP: Established ledger of type {type(self._ledger)} and size "
                      f"[{self._historyTime} x 18]")

        if poList is None:
            poList = []
            logging.debug(f"Firm {self._id}, t=SETUP: Created empty list poList")
        self.poList = poList

        self._producedPos = []
        logging.debug(f"Firm {self._id}, t=SETUP: Created empty list producedPos")

        self._productionOrder = 0
        logging.debug(f"Firm {self._id}, t=SETUP: Set initial productionOrder to {self._productionOrder}")

        self._productionQueue = []
        logging.debug(f"Firm {self._id}, t=SETUP: Created empty list productionQueue")

        self._shippedPos = []
        logging.debug(f"Firm {self._id}, t=SETUP: Created empty list shippedPos")

        self._shippingQueue = []
        logging.debug(f"Firm {self._id}, t=SETUP: Created empty list shippingQueue")

        if supplierList is None:
            supplierList = []
            logging.debug(f"Firm {self._id}, t=SETUP: Created empty list supplierList")
        self.supplierList = supplierList

        # Error checking has already been performed on historyTime
        self._timeIndex = historyTime
        logging.debug(f"Firm {self._id}, t=SETUP: Set initial timeIndex to {self._timeIndex}")

        self._wipInventory = wipInventory
        if not isinstance(wipInventory, int):
            message = "wipInventory must be an int"
            logging.error(message)
            raise TypeError(message)
        if wipInventory < 0:
            message = "wipInventory value cannot be less than zero"
            logging.error(message)
            raise Exception(message)

        logging.info(f"Firm {self._id}, t=SETUP: Initialization is complete.")

    # Getters and setters
    @property
    def customerList(self):
        return self._customerList

    @customerList.setter
    def customerList(self, value):
        if not value:
            self._customerList = []
            logging.debug(f"Firm {self._id}, t=SETUP: Created empty list customerList")
        else:
            self._customerList = value
            logging.debug(f"Firm {self._id}, t=SETUP: Created customerList with values {value} ")
            if not isinstance(value, list):
                message = "customerList must be a list object"
                logging.error(message)
                raise TypeError(message)
            for customer in value:
                if not isinstance(customer, int):
                    message = "Customer value in customerList must be an integer"
                    logging.error(message)
                    raise TypeError(message)
        if self._timePeriod > 0:
            logging.warning(f"Firm {self._id}, t={self._timePeriod}: customerlist changed after the simulation has "
                            f"started to run")

    @property
    def customerPoList(self):
        return self._customerPoList

    @customerPoList.setter
    def customerPoList(self, value):
        if not value:
            self._customerPoList = []
            logging.debug(f"Firm {self._id}, t={self._timePeriod}: Created empty list customerPoList")
        else:
            self._customerPoList = value
            logging.debug(f"Firm {self._id}, t={self._timePeriod}: Created customerPoList with values {value}")
            if not isinstance(value, list):
                message = "customerPoList must be a list object"
                logging.error(message)
                raise TypeError(message)
            for po in value:
                if not isinstance(po, PO.PO):
                    message = "Objects in customerPoList must be PO objects"
                    logging.error(message)
                    raise TypeError(message)

    @property
    def historyTime(self):
        return self._historyTime

    @property
    def id(self):
        return self._id

    @property
    def poList(self):
        return self._poList

    @poList.setter
    def poList(self, value):
        if not value:
            self._poList = []
            logging.debug(f"Firm {self._id}, t={self._timePeriod}: Created empty list poList")
        else:
            self._poList = value
            logging.debug(f"Firm {self._id}, t={self._timePeriod}: Created poList with values {value}")
            if not isinstance(value, list):
                message = "poList must be a list object"
                logging.error(message)
                raise TypeError(message)
            for po in value:
                if not isinstance(po, PO.PO):
                    message = "Objects in poList must be PO objects"
                    logging.error(message)
                    raise TypeError(message)

    @property
    def productionQueue(self):
        return self._productionQueue

    @property
    def shipDelay(self):
        return self._shipDelay

    @property
    def shippingQueue(self):
        return self._shippingQueue

    @property
    def supplierList(self):
        return self._supplierList

    @supplierList.setter
    def supplierList(self, value):
        if not value:
            self._supplierList = []
            logging.debug(f"Firm {self._id}, t={self._timePeriod}: Created empty list supplierList")
        else:
            self._supplierList = value
            logging.debug(f"Firm {self._id}, t={self._timePeriod}: Created supplierList with values {value}")
            if not isinstance(value, list):
                message = "supplierList must be a list object"
                logging.error(message)
                raise TypeError(message)
            for supplier in value:
                if not isinstance(supplier, int):
                    message = "Supplier value in supplierList must be an integer"
                    logging.error(message)
                    raise TypeError(message)
        if self._timePeriod > 0:
            logging.warning("supplierList changed after the simulation has started to run")

    def actualizeCost(self, moneyCost, co2Cost, waterCost):
        """
        Records this period’s cost for acreage + trucking.

        :param float moneyCost: total $ cost incurred for this firm at current timeIndex
        :param float co2Cost:   total lbs CO₂ emitted at this timeIndex
        :param float waterCost: total gallons H₂O used at this timeIndex
        """
        idx = self._timeIndex
        self._actualizedCostMoneyArray[idx] = moneyCost
        self._actualizedCostCO2Array[idx]   = co2Cost
        self._actualizedCostWaterArray[idx] = waterCost
        logging.debug(
            f"Firm {self._id}, t={self._timePeriod}, f=actualizeCost(): "
            f"Money=${moneyCost:.2f}, CO2={co2Cost:.2f} lbs, Water={waterCost:.2f} gal"
        )

    
    def actualizeDemand(self, amount):
        """
        Records the time period's demand information.

        :param int amount:
            The demand for the current time period.
        :return:
        """
        self._actualizedDemandArray[self._timeIndex] += amount
        self._ledger[self._timeIndex, 17] = amount  # actual_demand column
        logging.debug(f"Firm {self._id}, t={self._timePeriod}, f=actualizeDemand():  Updated actualizedDemandArray and "
                      f"Ledger to value {amount}")

    def beginningOfDay(self):
        """
        Basic function, whose primary purpose is to be overriden by the wholesaler class
        :return:
        """
        pass

    def calculateFutureDemand(self, smoothingValue):
        """
        A function to calculate future demand predictions. This function incorporates demand smoothing, which has been
        shown in the literature to reduce variability in orders. Demand smoothing is implemented particularly by taking
        the average of the current time period's forecasted and the current time period's actual demand.

        :param int smoothingValue:
            smoothingValue takes a value of 1 or 2.
            Choosing 1 turns off smoothing (future demand = current time period demand).
            Choosing 2 turns on smoothing of 1 time period
                [future demand = (current demand + current forecasted demand) / 2]
        :return:
            An integer value for the calculated demand projection
        """
        if smoothingValue == 1:  # No smoothing
            newDemand = int(self._actualizedDemandArray[self._timeIndex])
            message = f"Firm {self._id}, t={self._timePeriod}, f=calculateFutureDemand(): Smoothing off; future demand " \
                      f"equals current time period demand: {newDemand}"
        elif smoothingValue == 2:  # Smoothing on
            value1 = self._actualizedDemandArray[self._timeIndex]
            value2 = self._demandForecastArray[self._timeIndex]
            newDemand = int((value1 + value2) / 2)
            message = f"Firm {self._id}, t={self._timePeriod}, f=calculateFutureDemand(): A demand projection of " \
                      f"{newDemand} was calculated by" \
                      f"averaging current demand ({value1}) and the current time's forecast ({value2})"
        else:
            message = f"Firm {self._id}, t={self._timePeriod}, f=calculateFutureDemand(): Smoothing value chosen is not " \
                      f"recognized"
            logging.error(message)
            raise Exception(message)
        logging.debug(message)
        return newDemand

    def calculateHistoricalDemand(self, demandMu):
        """
        Basic function, whose primary purpose is to be overridden by the Manufacturer class

        :param int demandMu:
            A parameter representing the average final customer demand in the simulation
        :return: simulation average demand
        """
        return demandMu

    def calculateSalesDuringDelay(self):
        """
        A function to calculate the amount of sales that will occur during the time that one WIP order is being shipped

        :return:
            calculated sales during delay
        """
        salesDuringDelay = 0
        endTime = self._shipDelay - 1
        for time in range(endTime):
            salesDuringDelay += self._demandForecastArray[self._timePeriod + self._shipDelay + time]
        message = f"Firm {self._id}, t={self._timePeriod}, f=calculateSalesDuringDelay(): A value for salesDuringDelay " \
                  f"was calculated as {salesDuringDelay} by adding up {endTime} future time periods"
        logging.debug(message)
        return salesDuringDelay

    def calculateSupplyInTransit(self):
        """
        A function to calculate the amount of WIP that is still in transit, expected to be received. This value is used
        when making ordering decisions.

        :return:
            calculated amount of supply currently in transit
        """
        supplyInTransit = 0
        count = 0
        self.warningLoop(self.poList, "calculateSupplyInTransit()")
        for po in self.poList:
            if po.arrivalTime > self._timePeriod and not po.customerClosed:
                supplyInTransit += po.fulfilledAmt  # Assumes that the customer knows they were shorted when
                # product ships
                count += 1  # For logging purposes
        logging.debug(
            f"Firm {self._id}, t={self._timePeriod}, f=calculateSupplyInTransit(): A value of {supplyInTransit} "
            f"was calculated for the total supply in transit, with {count} PO's outstanding.")
        return supplyInTransit

    def calculateSupplyOrder(self):
        """
        A function that will calculate the total amount of WIP required, taking into account desired WIP, current WIP,
        how many sales are expected over the shipping delay time, and the WIP already in transit.

        :return:
            An integer value for the total amount of WIP required
        """
        salesDuringDelay = self.calculateSalesDuringDelay()
        supplyInTransit = self.calculateSupplyInTransit()
        wipInventory = self._wipInventory
        desiredWipInventory = self._desiredWipInventory
        order = max(0, int(desiredWipInventory +  # Desired WIP including safety stock
                           salesDuringDelay -  # Expected sales over the shipping delay period
                           supplyInTransit -  # Inventory expected on the way
                           wipInventory))  # Current WIP inventory
        self._ledger[self._timeIndex, 14] = supplyInTransit  # wip_in_transit column
        logging.debug(f"Firm {self._id}, t={self._timePeriod}, f=calculateSupplyOrder(): A value of {order} was "
                      f"calculated for the supply order"
                      f"and was logged in the ledger, calculated as\n desiredWipInventory ({desiredWipInventory}) + "
                      f"salesDuringDelay ({salesDuringDelay}) - supplyInTransit ({supplyInTransit}) - wipInventory "
                      f"({wipInventory})")
        return order

    def chooseSuppliers(self, currentTimePeriod):
        """
        A basic function whose primary purpose is to be overriden by the Wholesaler class.

        :param currentTimePeriod:
            The current simulation time period
        :return:
        """
        pass

    def createHistory(self, custPos, supPos):
        """
        Takes in two lists of po objects and adds it to internal lists.

        :param list custPos:
            A list of PO objects that are originated from customers
        :param list supPos:
            A list of PO objects that are originated by the focal firm, sent to suppliers
        :return:
        """
        count1 = 0
        count2 = 0
        self.warningLoop(custPos, "createHistory()")
        for po in custPos:
            self._customerPoList.append(po)
            count1 += 1
        self.warningLoop(supPos, "createHistory()")
        for po in supPos:
            self._poList.append(po)
            count2 += 1
        logging.debug(f"Firm {self._id}, t={self._timePeriod}, f=createHistory(): Appended {count1} historical PO's to "
                      f"customerPOList and {count2} historical PO's to poList.")

    def createPo(self, supplier, orderAmt, orderTime):
        """
        Creates a new PO object

        :param int supplier:
            Unique firm identifier for supplier
        :param int orderAmt:
            Amount ordered
        :param int orderTime:
            Time period order was placed
        :return:
            A list object (which is mutable) containing the purchase order, which allows the PO to be changed later
        """
        firmId = self._id
        purchaseOrder = PO.PO(firmId, supplier, orderAmt, orderTime)
        self.poList.append(purchaseOrder)
        length = len(self.poList)
        logging.debug(
            f"Firm {firmId}, t={self._timePeriod}, f=createPo(): Created a new PO with supplier {supplier} and "
            f"amount {orderAmt}, in time period {orderTime}. Appended PO to poList, now of length {length}")
        return [purchaseOrder]

    def createProductionOrder(self):
        """
        A method that counts up the order quantities for each customer PO and then puts that integer value into the
        production queue.

        :return:
        """
        productionAmount = 0
        wipInventory = self._wipInventory
        self.warningLoop(self.productionQueue, "createProductionOrder()")  # Used for simulation performance purposes
        for queuePo in self.productionQueue:
            productionAmount += queuePo.orderAmt
        if productionAmount > wipInventory:
            logging.warning(
                f"Firm {self._id}, t={self._timePeriod}, f=createProductionOrder(): Not enough WIP inventory "
                f"to produce desired amount.\nWanted to produce {productionAmount}, only able to produce "
                f"{wipInventory}")
            productionAmount = wipInventory
        if productionAmount < 0:
            raise Exception("Production order cannot be less than 0")
        self._productionOrder = productionAmount
        self._ledger[self._timeIndex, 9] = productionAmount  # total_new_production_orders column
        logging.debug(f"Firm {self._id}, t={self._timePeriod}, f=createProductionOrder(): Created production order for "
                      f"{productionAmount} units and logged in the ledger.")

    def createSupplyOrders(self, amount):
        """
        A function that will create PO objects based on the desired supply.

        :param amount:
            The amount of WIP supply to be ordered.
        :return:
            A list of PO objects to be sent to the supplier.
        """
        newPoList = []
        count = 0
        for supplier in self.supplierList:
            newPo = self.createPo(supplier, amount, self._timePeriod)
            newPoList.append(newPo[0])
            count += 1
        logging.debug(f"Firm {self._id}, t={self._timePeriod}, f=createSupplyOrders: Created {count} new supply orders "
                      f"of amount {amount}.")
        return newPoList

    def endOfDay(self):
        """
        Records information in ledger, cleans out PO lists, advances timePeriod and timeIndex.

        :return:
        """
        # End of day error checking
        if self._wipInventory < 0:
            logging.error(f"Firm {self._id}, t={self._timePeriod}, f=endOfDay: WIP inventory has become negative.")
            raise Exception("WIP inventory has become negative")
        if self._fgInventory < 0:
            logging.error(f"Firm {self._id}, t={self._timePeriod}, f=endOfDay: FG inventory has become negative.")
            raise Exception("FG inventory has become negative")
        wipMaterialsUsed = 0
        self.warningLoop(self.productionQueue, "endOfDay()")
        for po in self.productionQueue:
            wipMaterialsUsed += po.fulfilledAmt
        self._ledger[self._timeIndex, 4] = wipMaterialsUsed  # WIP Materials Used in Production
        self._ledger[self._timeIndex, 10] = wipMaterialsUsed  # Finished goods produced
        # Clears production queue only if it is closed by the supplier
        self.productionQueue[:] = [po for po in self.productionQueue if not po.supplierClosed]
        queueLength = len(self.productionQueue)
        if queueLength > 0:
            logging.warning(
                f"Firm {self._id}, t={self._timePeriod}, f=endOfDay: Attempted to clear production queue, but "
                f"{queueLength} PO's remain.")
        self._ledger[self._historyTime + self._timePeriod, 0] = self._timePeriod
        # Input all the ending values for current time period
        self._ledger[self._timeIndex, 5] = self._wipInventory
        self._ledger[self._timeIndex, 13] = self._fgInventory
        self._ledger[self._timeIndex, 16] = self._demandForecastArray[self._timeIndex]
        # Clean up some random issues
        # (If total_wip_received in current time is 0 and production_wip_inventory is 0)
        if self._ledger[self._timeIndex, 2] == 0 and self._ledger[self._timeIndex, 3] == 0:
            # This issue arises when no po's are received in the current time period
            # (Set the production_wip_inventory to the beginning_wip_inventory
            self._ledger[self._timeIndex, 3] = self._ledger[self._timeIndex, 1]
        logging.debug(f"Firm {self._id}, t={self._timePeriod}, f=endOfDay: Finished updating ledger")
        # Move closed POs to a new list
        count = 0
        self.warningLoop(self.customerPoList, "endOfDay()")
        for po in self.customerPoList:
            if po.closed:
                self._closedCustomerPos.append(po)
                count += 1
        self.warningLoop(self.poList, "endOfDay()")
        for po in self.poList:
            if po.closed:
                self._closedSupplierPos.append(po)
                count += 1
        self.customerPoList[:] = [po for po in self.customerPoList if not po.closed]
        self.poList[:] = [po for po in self.poList if not po.closed]
        logging.debug(f"Firm {self._id}, t={self._timePeriod}, f=endOfDay: Moved {count} closed PO's to new list."
                      f"customerPoList length: {len(self.customerPoList)}. poList length: {len(self.poList)}")
        # Advance the time period
        self._timePeriod += 1
        self._timeIndex += 1
        # Input all the beginning values for next time period
        if self._timeIndex < self._totalTime:
            self._ledger[self._timeIndex, 1] = self._wipInventory
            self._ledger[self._timeIndex, 7] = self._fgInventory
        logging.debug(f"Firm {self._id}, t={self._timePeriod}, f=endOfDay: Advanced internal timePeriod.")

    def initializeAgent(self, historicalWeeklyDemand):
        """
        Sets up the parameters, attributes and data structures for use in the simulation

        :param int historicalWeeklyDemand:
            The value that will be used for each week's historical demand.
        :return:
        """
        self.setClassAttributes(historicalWeeklyDemand)
        self.setHistoricalLedger(historicalWeeklyDemand)
        self.setForecast(historicalWeeklyDemand)
        self.setDemandArray(historicalWeeklyDemand)
        self.setHistoricalReturnArray(historicalWeeklyDemand)
        self.timeZeroSetup()

    def logSupplyPo(self, poNumber, timeIndex):
        """
        Basic function, whose primary purpose is to be overridden by the Wholesaler class
        :param int poNumber:
            Unique identifier for PO
        :param int timeIndex:
            The current time period for the simulation. Primarily used for data arrays that do not have historical data.
        :return:
        """
        pass

    def makeProduct(self, poToMake, amount):
        """
        A function to represent a real-life batch order for production.
        It is assumed that no product is lost to spoilage, waste, or defects. That is to say, every firm is assumed to
        be 100% efficient on the transfer from WIP to FG. It is also assumed that there is a 1:1 relationship between
        WIP and FG (i.e. that every finished good only takes one WIP to make the product).

        :param list poToMake:
            a list with a single PO object that is to be fulfilled. A list is passed so that the PO object can be
            changed without creating a new object.
        :param int amount:
            The amount of product to be produced. Can take on values 0 < amount <= orderedAmount
        :return:
        """
        poToMake[0].updatePO({'fulfilledAmt': amount,
                              'fulfilledTime': self._timePeriod})
        self._fgInventory += amount
        self._wipInventory -= amount
        logging.debug(f"Firm {self._id}, t={self._timePeriod}, f=makeProduct: Created {amount} new product.")

    def production(self):
        """
        A function to represent the real-life production decision process. The function looks at its current WIP inventory
        and determines which and how much finished goods to produce.

        The function will loop through the productionQueue one at a time and determine if there is enough WIP inventory
        to fulfill the order. The simulation allows partial fulfillment of orders, but only for the current time period.
        Whatever is not produced in the current time period for a particular PO is cancelled. In other words, a supplier
        cannot partially fulfill an order in t and then fulfill the rest in t+1

        :return:
        """
        leftToProduce = self._productionOrder
        self.warningLoop(self.productionQueue, "production()")
        random.shuffle(self.productionQueue)  # Randomize the order of the queue so one customer doesnt always get prod
        for po in self.productionQueue:  # Go through the production PO queue one at a time and try to make it
            if leftToProduce < 0:
                message = f"Firm {self._id}, t={self._timePeriod}, f=production(): leftToProduce value is less than 0."
                logging.error(message)
                raise Exception(message)
            po.supplierClosed = True  # All orders are closed in the time period they are received
            if leftToProduce == 0:
                po.updatePO({'fulfilledAmt': 0,
                             'fulfilledTime': self._timePeriod})
                """
                A potential change here: if no product is made, close the po?
                """
                self.shippingQueue.append(po)  # Appending to shipping queue but not produced POs
                logging.warning(f"Firm {self._id}, t={self._timePeriod}, f=production(): Unfulfilled PO moved to "
                                f"shippingQueue.")
                continue  # Move to next PO
            elif po.orderAmt <= leftToProduce:  # Can produce the full amount
                leftToProduce -= po.orderAmt
                self.makeProduct([po], po.orderAmt)
            else:  # Can produce a partial (but not 0) amount
                self.makeProduct([po], leftToProduce)
                leftToProduce = 0
            self._producedPos.append(po)
            self.shippingQueue.append(po)
        logging.debug(f"Firm {self._id}, t={self._timePeriod}, f=production(): Production complete, with "
                      f"{leftToProduce} finished products still to produce.")
        self._ledger[self._timeIndex, 11] = self._fgInventory  # fg_inventory_after_production column

    def orderSupplies(self, currentTimePeriod):
        """
        A method to represent the real-life process of ordering supplies for expected future demand.

        :param currentTimePeriod:
            The current simulation time period.
        :return:
        """
        self.chooseSuppliers(currentTimePeriod)
        newOrder = self.calculateSupplyOrder()
        self._ledger[self._timeIndex, 6] = newOrder  # total_wip_ordered column
        if newOrder == 0:
            orderList = []
            logging.warning(f"Firm {self._id}, t={self._timePeriod}, f=orderSupplies: No WIP order placed.")
        else:
            orderList = self.createSupplyOrders(newOrder)
            logging.debug(f"Firm {self._id}, t={self._timePeriod}, f=orderSupplies: {newOrder} supplies ordered.")
        return orderList

    def receiveCustomerDemand(self, customerPOs):
        """
        A method to represent the real-life process of receiving customer orders. Turns customer orders into production
        orders.

        :param list customerPOs:
            A list of PO objects representing all customer orders for the time period.
        :return:
        """
        amount = 0
        count = 0
        self.warningLoop(customerPOs, "receiveCustomerDemand()")
        for po in customerPOs:
            self.customerPoList.append(po)
            self.productionQueue.append(po)
            amount += po.orderAmt
            count += 1
        logging.debug(f"Firm {self._id}, t={self._timePeriod}, f=receiveCustomerDemand(): Created {count} PO's for a "
                      f"total amount of {amount} product")
        self.createProductionOrder()
        self.actualizeDemand(amount)

    def receiveWipOrder(self, poNumber):
        """
        Intakes the WIP order into inventory, checks to see if the PO has been fulfilled

        :param int poNumber:
            Unique identifier for PO
        """
        totalWipReceived = 0
        warn = True
        self.warningLoop(self.poList, "receiveWipOrder()")
        for po in self.poList:
            if po.id == poNumber:
                self._wipInventory += po.fulfilledAmt
                totalWipReceived += po.fulfilledAmt
                po.customerClosed = True  # PO needs to be closed by both supplier and customer
                warn = False
        self._ledger[self._timeIndex, 2] += totalWipReceived  # total_wip_received column
        self._ledger[self._timeIndex, 3] = self._wipInventory  # production_wip_inventory column
        logging.debug(f"Firm {self._id}, t={self._timePeriod}, f=receiveWipOrder(): Received {totalWipReceived} WIP "
                      f"product.")
        self.logSupplyPo(poNumber, self._timeIndex)
        if len(self.poList) == 0:
            logging.warning(f"Firm {self._id}, t={self._timePeriod}, f=receiveWipOrder(): Trying to receive WIP order# "
                            f"{poNumber} but internal PO list is empty.")
        elif warn:
            logging.warning(f"Firm {self._id}, t={self._timePeriod}, f=receiveWipOrder(): Trying to receive WIP order# "
                            f"{poNumber} but that PO# cannot be found")

    def resetFirm(self, demandMu):
        """
        Resets the firm to default values, in preparation for a new simulation.

        :param int demandMu:
            A parameter representing the average final customer demand in the simulation
        :return:
        """
        # ~~~~~~~~~~~~~~~~
        self.setClassAttributes(demandMu)
        # This needs to be updated for the wholesaler class
        # ~~~~~~~~~~~~~~~~
        self._actualizedDemandArray = np.zeros(self._totalTime, dtype=int)

        self._actualizedCostMoneyArray = np.zeros(self._totalTime, dtype=float)
        self._actualizedCostCO2Array   = np.zeros(self._totalTime, dtype=float)
        self._actualizedCostWaterArray = np.zeros(self._totalTime, dtype=float)

        # alpha not reset
        self._closedCustomerPos = []
        self._closedSupplierPos = []
        # customerList not reset
        self.customerPoList = []
        self._demandForecastArray = np.zeros(self._totalTime + self._shipDelay, dtype=int)

        # historyTime not reset
        # id not reset
        self._ledger = np.zeros((self._historyTime, 18), dtype=int)
        self.poList = []
        self._producedPos = []
        self._productionOrder = 0
        self._productionQueue = []
        # runTime not reset
        # shipDelay not reset
        self._shippedPos = []
        self._shippingQueue = []
        # supplierList not reset
        self._timeIndex = self._historyTime
        self._timePeriod = 0
        # totalTime not reset

        # Sets desiredWipInventory, fgInventory, and wipInventory
        self.setClassAttributes(demandMu)
        logging.info(f"Firm {self._id}, t={self._timePeriod}, f=resetFirm(): Firm reset.")

    def sendCustomerShipments(self, timePeriod):
        """
        A method to represent the real-life process of sending customer shipments. Updates the PO with the shipped time,
        calculates arrival time and updates PO accordingly. Empties shipping queue when all done.

        :param int timePeriod:
            Current simulation time period
        :return:
        """
        shippedAmount = 0
        count = 0
        if len(self.shippingQueue) == 0:
            logging.warning(f"Firm {self._id}, t={self._timePeriod}, f=sendCustomerShipments(): No items shipped.")
            return False
        else:
            self.warningLoop(self.shippingQueue, "sendCustomerShipments()")
            for po in self.shippingQueue:
                po.updatePO({'arrivalTime': timePeriod + self._shipDelay})
                po.supplierClosed = True  # Make sure the PO is closed
                self._fgInventory -= po.fulfilledAmt
                shippedAmount += po.fulfilledAmt
                self._shippedPos.append(po)
                count += 1
            self._shippingQueue = []  # Every time period empty out the shipping queue
        self._ledger[self._timeIndex, 12] = shippedAmount  # total_fg_shipped column
        logging.debug(f"Firm {self._id}, t={self._timePeriod}, f=sendCustomerShipments(): {count} customer PO's sent"
                      f"with {shippedAmount} total product.")
        return True

    def sendData(self, option):
        """
        Returns firm-level data to be aggregated with firm-level data for other firms.

        :param int option:
            Takes two values, 'Ledger' or 'Demand'. Ledger sends the entire ledger, while demand sends only the
            demand array.
        :return: Specified data in ndArray form
        """
        if option == 'Ledger':
            logging.info(f"Firm {self._id}, t={self._timePeriod}, f=sendData(): Sent ledger data to simulation")
            return self._ledger
        elif option == 'Demand':
            logging.info(f"Firm {self._id}, t={self._timePeriod}, f=sendData(): Sent demand data to simulation")
            return self._actualizedDemandArray        
        elif option == 'CostMoney':
            return self._actualizedCostMoneyArray
        elif option == 'CostCO2':
            return self._actualizedCostCO2Array
        elif option == 'CostWater':
            return self._actualizedCostWaterArray
        else:
            message = f"Firm {self._id}, t={self._timePeriod}, f=sendData(): Unknown option chosen."
            logging.error(message)
            raise Exception(message)

    def setClassAttributes(self, historicalWeeklyDemand):
        """
        Sets (or resets) the initial class attributes

        :param int historicalWeeklyDemand:
            The value that will be used for each week's historical demand.
        :return:
        """
        self._desiredWipInventory = int(historicalWeeklyDemand * (1 + self._alpha))
        self._wipInventory = int(historicalWeeklyDemand * self._alpha)
        self._fgInventory = 0
        logging.debug(f"Firm {self._id}, t={self._timePeriod}, f=setClassAttributes(): Created initial class "
                      f"attributes")

    def setDemandArray(self, historicalWeeklyDemand):
        """
        A method that will set the historical demand values for the demand array

        :param int historicalWeeklyDemand:
            The value that will be used for each week's historical demand.
        :return:
        """
        for time in range(self._historyTime):
            self._actualizedDemandArray[time] = historicalWeeklyDemand
        logging.debug(f"Firm {self._id}, t={self._timePeriod}, f=setDemandArray(): Set historical values in demand "
                      f"array")

    def setForecast(self, historicalWeeklyDemand):
        """
        Sets up the preliminary demand forecast

        :param int historicalWeeklyDemand:
            The value that will be used for each week's historical demand.
        :return: None
        """
        for time in range(self._totalTime):
            self._demandForecastArray[time] = historicalWeeklyDemand
        logging.debug(f"Firm {self._id}, t={self._timePeriod}, f=setForecast(): Created initial demand forecast array")

    def setHistoricalLedger(self, historicalWeeklyDemand):
        """
        A method that will establish the historical time period values for the firm's internal ledger

        :param int historicalWeeklyDemand:
            The value that will be used for each week's historical demand.
        :return:
        """
        for time in range(self._historyTime):
            if time == 0:
                self._ledger[-time, 0] = -self._historyTime  # time column
            else:
                self._ledger[-time, 0] = -time  # time column
            self._ledger[time, 1] = self._wipInventory  # beginning_wip_inventory
            self._ledger[time, 2] = historicalWeeklyDemand  # total_wip_received
            self._ledger[time, 3] = historicalWeeklyDemand + self._wipInventory  # production_wip_inventory
            self._ledger[time, 4] = historicalWeeklyDemand  # wip_used_in_production
            self._ledger[time, 5] = self._wipInventory  # ending_wip_inventory
            self._ledger[time, 6] = historicalWeeklyDemand  # total_wip_ordered
            self._ledger[time, 7] = self._fgInventory  # beginning_fg_inventory
            self._ledger[time, 9] = historicalWeeklyDemand  # total_new_production_orders
            self._ledger[time, 10] = historicalWeeklyDemand  # total_fg_produced
            self._ledger[time, 11] = historicalWeeklyDemand  # fg_inventory_after_production
            self._ledger[time, 12] = historicalWeeklyDemand  # total_fg_shipped
            self._ledger[time, 13] = self._fgInventory  # ending_fg_inventory
            self._ledger[time, 14] = historicalWeeklyDemand * (self._shipDelay - 1)  # wip_in_transit
            self._ledger[time, 15] = self._desiredWipInventory  # desired_wip_inventory
            self._ledger[time, 16] = historicalWeeklyDemand  # forecasted_demand
            self._ledger[time, 17] = historicalWeeklyDemand  # actual_demand
        # Add on the extra rows for the simulation time
        rows = np.zeros((self._runTime, 18), dtype=int)
        self._ledger = np.vstack((self._ledger, rows))
        logging.debug(f"Firm {self._id}, t={self._timePeriod}, f=setHistoricalLedger(): Set historical data in ledger")

    def setHistoricalReturnArray(self, historicalWeeklyDemand):
        """
        Basic function, whose primary purpose is to be overriden by the Wholesaler class

        :param int historicalWeeklyDemand:
            The value that will be used for each week's historical demand.
        :return:
        """
        pass

    def timeZeroSetup(self):
        """
        Sets the initial class variables for beginning of time zero

        :return:
        """
        self._ledger[self._timeIndex, 1] = self._wipInventory  # beginning_wip_inventory
        self._ledger[self._timeIndex, 7] = self._fgInventory  # beginning_fg_inventory
        self._ledger[self._timeIndex, 15] = self._desiredWipInventory  # desired_wip_inventory
        logging.debug(f"Firm {self._id}, t={self._timePeriod}, f=timeZeroSetup(): Established beginning of time period "
                      f"data")

    def updateDemandForecast(self, smoothingValue):
        """
        A function to represent the real-life process of demand  forecasting for a firm.
        A call to calculateFutureDemand(...) is made and an int representing the single-time-period demand forecast is
        returned. desiredWipInventory (a function of alpha and futureDemand) is updated and then recorded in the ledger.
        forecasted demand is then updated in the demandForecastArray for all times: [t+1, t+shipDelay+1]

        :param int smoothingValue:
            smoothingValue takes a value of 1 or 2.
            Choosing 1 turns off smoothing (future demand = current time period demand).
            Choosing 2 turns on smoothing of 1 time period
                [future demand = (current demand + current forecasted demand) / 2]
        :return:
        """
        newDemand = self.calculateFutureDemand(smoothingValue)
        self._desiredWipInventory = int(newDemand * (1 + self._alpha))
        if self._timeIndex + 1 < self._totalTime:
            self._ledger[self._timeIndex + 1, 15] = self._desiredWipInventory  # desired_wip_inventory column
        count = 0
        # Prevent index out of bound error
        if self._timeIndex + self._shipDelay + 1 >= len(self._demandForecastArray):
            endTime = len(self._demandForecastArray)
        else:
            endTime = self._timeIndex + self._shipDelay + 1
        # Update demand forecast for future time periods [t + 1, t + shipDelay + 1]
        for t in range(self._timeIndex + 1, endTime):
            self._demandForecastArray[t] = newDemand
            count += 1
        logging.debug(f"Firm {self._id}, t={self._timePeriod}, f=updateDemandForecast(): Updated {count} total demand "
                      f"forecasts with demand value of {newDemand}. Changed desiredWipInventory to "
                      f"{self._desiredWipInventory}.")

    def warningLoop(self, listToCheck, funcName):
        """
        A function used for logging purposes. Will check the length of a list that is about to be looped through.
        Will output a warning in the log if the list is longer than 10. This helps to identify potential causes of
        long processing times.

        :param list listToCheck:
            A list object that is about to be looped through
        :param string funcName:
            The name of the function that the warningLoop was called in

        :return:
        """
        length = len(listToCheck)
        if length > 10:
            logging.warning(
                f"Firm {self._id}, t={self._timePeriod}, f=warningLoop(): Looping through a PO list of size "
                f"{length} in function {funcName}")


class Retailer(Firm):
    """
    A subclass that inherits from Firm3, representing a retailer (level 1 in supply chain).

    Currently does not override any methods.
    """

    def __init__(self,
                 alpha,
                 runTime,
                 historyTime,
                 shipDelay,
                 idNum,
                 customerList=None,
                 customerPoList=None,
                 fgInventory=0,
                 supplierList=None,
                 poList=None,
                 wipInventory=0,
                 ):
        super().__init__(alpha,
                         runTime,
                         historyTime,
                         shipDelay,
                         idNum,
                         customerList,
                         customerPoList,
                         fgInventory,
                         supplierList,
                         poList,
                         wipInventory
                         )


class Wholesaler(Firm):
    """
    A subclass that inherits from Firm3, representing a wholesaler (level 2 in supply chain).

    Instance Attributes (Unique to Wholesaler):
    ---------
    covarianceMatrix: np array
        A 2x2 array representing the covariance in return between firm 2 and 3, both suppliers for firm 1
    fulfilledBySupplier:
        An s x t list, where s = number of suppliers and t = number of time periods.
        Splits out amount actually fulfilled by particular supplier.
    historicalPortfolio:
        An s x t list, where s = number of suppliers and t = number of time periods.
        Keeps track of each time period's portfolio values.
    historicalReturnRuleOption:
        Sets how historical returns should be treated. If a supplier does not receive an order, how is that treated when
        calculating historical return?
        Option 1: Treated as a 0
        Option 2: Treated as a 1
        Option 3 (Default): To avoid such a decision, choosing this option will mean that the wholesaler uses the last n
            timeperiods where an order was sent to that supplier. i.e., It will ignore time periods where order was 0.
    numTimePeriodsForCalc: int
        The number of previous time periods that the wholesaler will use to calculate historical return, which is
        used to optimize their portfolio
    ordersBySupplier: list
        An s x t list, where s = number of suppliers and t = number of time periods.
        Splits out supply orders by particular supplier.
    returnBySupplier:
        An s x t list, where s = number of suppliers and t = number of time periods.
        Splits out the return calculation by particular supplier.
    returnBySupplierForCalc:
        A list subsetted from returnBySupplier. How this is populated depends on historicalReturnRuleOption,
        and numTimePeriodsForCalc. For example, if numTimePeriodsForCalc is 5, the list is s x 5, where s = number of
        suppliers.
    riskTolerance: float
        A parameter used in the optimization calculation.  A higher value will place more emphasis on rewards at
        a higher risk, whereas lower values emphasize reducing risk at the expense of lower rewards. It is equivalent
        to desired return, therefore must take on values of [0,1].
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

    Instance Methods (Unique to Wholesaler)
    ---------
    calculateCovariance(timePeriods):
        A method that will calculate the covariance for the historical return.
    calculateWipOrder(amount):
        A method that will take the total amount of WIP required and splits it into an order amount for each supplier
    checkCovarianceMatrixError():
        Not currently implemented
    setPortfolio(currentTimePeriod):
        Sets the instance variable wholesalerProfile to the given value.
    setReturnBySupplierForCalc(timeIndex):
        Checks to see if the array should be updated. It will not be updated for a specific firm if there was no order
        placed for that firm in the time period.
        Performed at the beginning of the timePeriod specified as a passed parameter for use in the next timePeriod.
        Checks historicalReturnRuleOption:
        Option 1: Treated as a 0
        Option 2: Treated as a 1
        Option 3 (Default): To avoid such a decision, choosing this option will mean that the wholesaler uses the last n
            timeperiods where an order was sent to that supplier. i.e., It will ignore time periods where order was 0.
    solveProgram(currentTimePeriod):
        Ensures that the portfolio cannot be changed until after historical time is complete.
    solveOptimize(currentTimePeriod):
        Solves an optimization problem using the Markowitz portfolio allocation model.
        (H Markowitz. Portfolio Selection: Efficient Diversification of Investment. New York: John Wiley & Sons, 1959.)
        The model is:
        minimize:    W^T*C*W
        subject to: W^T*1(vector) = 1
                    W^T*E = Mu

        Where W is the vector of decision weights, C is the covariance matrix of returns, E is a vector of expected
        returns, and Mu is a measure of risk tolerance. Return is defined for our purposes to be:
         number of WIP received / number of WIP ordered (by supplier).
        Expected return is defined to be the historical average of return, for t-time periods.


    Instance Methods (Which override Firm3 method)
    ---------
    beginningOfDay()
        A function for those aspects of the simulation that the firm needs to finish before the actions in the current
        time period (but after all other firms have finished their previous time period). For actions that do not
        rely on other firms, those methods should go in the endOfDay function.

        A supplier will produce an item the same "time period", but proceeding the actions of the customer. When the
        supplier produces the item, it then updates the PO with how much was actually fulfilled. The wholesaler needs
        this information to calculate its historical return. It therefore needs to be done beginning of day, not end
        of day. NOTE: currently the simulation does not add production time (orders are received, produced, and shipped
        the same day). If this changes, the way that historical return is calculated will have to change as well.
    chooseSuppliers(currentTimePeriod):
        The result of this function will be an update to the supplier portfolio.
    createSupplyOrders(orderAmount, currentTimePeriod):
        Overrides the Firm3 function to account for the wholesaler having two sources of supply. Creates PO objects
        representing the focal firms supply orders.
    logSupplyPo(poNumber, timeIndex)
        Overrides the Firm3 method to account for the wholesaler having two sources of supply. Logs data for each PO.
    setHistoricalReturnArray(historicalWeeklyDemand)
        Overrides the Firm3 function to account for the wholesaler having two sources of supply. Establishes historical
        data that is used in the simulation to optimize the portfolio.




    """

    def __init__(self,
                 alpha,
                 runTime,
                 historyTime,
                 shipDelay,
                 idNum,
                 customerList=None,
                 customerPoList=None,
                 fgInventory=0,
                 supplierList=None,
                 poList=None,
                 wipInventory=0,
                 wholesalerProfile=None,
                 wholesalerProfileOption='Default',
                 numTimePeriodsForCalc=5,
                 historicalReturnRuleOption=3,
                 covarianceMatrix=None,
                 riskTolerance=0.5
                 ):
        super().__init__(alpha,
                         runTime,
                         historyTime,
                         shipDelay,
                         idNum,
                         customerList,
                         customerPoList,
                         fgInventory,
                         supplierList,
                         poList,
                         wipInventory
                         )
        # Other error checking for covarianceMatrix is performed in Configs.py
        if covarianceMatrix is None:
            message = "Covariance matrix is not initialized."
            logging.error(message)
            raise Exception(message)
        self.covarianceMatrix = covarianceMatrix

        self._fulfilledBySupplier = np.zeros((self._totalTime, 2), dtype=int)
        logging.debug(
            f"Firm {self._id}, t=SETUP: Established fulfilledBySupplier of type {type(self._ledger)} and size "
            f"[{self._totalTime} x 2]")

        self._historicalPortfolio = np.zeros((self._totalTime, 2), dtype=float)
        logging.debug(
            f"Firm {self._id}, t=SETUP: Established historicalPortfolio of type {type(self._ledger)} and size "
            f"[{self._totalTime} x 2]")

        self._historicalReturnRuleOption = historicalReturnRuleOption
        if not (self._historicalReturnRuleOption == 1 or self._historicalReturnRuleOption == 2 or
                self._historicalReturnRuleOption == 3):
            message = "historicalReturnRuleOption must be 1, 2, or 3"
            logging.debug(message)
            raise Exception(message)

        self._numTimePeriodsForCalc = numTimePeriodsForCalc
        if self._numTimePeriodsForCalc < 0:
            message = "numTimePeriodsForCalc cannot be less than zero"
            logging.error(message)
            raise Exception(message)
        if not isinstance(self._numTimePeriodsForCalc, int):
            message = "numTimePeriodsForCalc must be an integer value"
            logging.error(message)
            raise TypeError(message)

        self._ordersBySupplier = np.zeros((self._totalTime, 2), dtype=int)
        logging.debug(
            f"Firm {self._id}, t=SETUP: Established ordersBySupplier of type {type(self._ledger)} and size "
            f"[{self._totalTime} x 2]")

        self._returnBySupplier = np.zeros((self._totalTime, 2), dtype=float)
        logging.debug(
            f"Firm {self._id}, t=SETUP: Established returnBySupplier of type {type(self._ledger)} and size "
            f"[{self._totalTime} x 2]")

        self._returnBySupplierForCalc = np.zeros((self._numTimePeriodsForCalc, 2), dtype=float)
        logging.debug(
            f"Firm {self._id}, t=SETUP: Established returnBySupplierForCalc of type {type(self._ledger)} and size "
            f"[{self._numTimePeriodsForCalc} x 2]")

        self.riskTolerance = riskTolerance
        if self.riskTolerance <= 0:
            message = "Risk tolerance value cannot be less than 0"
            logging.error(message)
            raise Exception(message)
        if self.riskTolerance > 1:
            message = "Risk tolerance value cannot be greater than 1. (Risk tolerance is equivalent to "
            "desired return, and return cannot be greater than 1."
            logging.error(message)
            raise Exception(message)

        if wholesalerProfile is None:
            message = "Wholesaler profile not initialized"
            logging.error(message)
            raise Exception(message)
        self.wholesalerProfile = wholesalerProfile
        if len(self.wholesalerProfile) != 2:
            message = "Length of wholesaler profile must be 2"
            logging.error(message)
            raise Exception(message)
        if sum(self.wholesalerProfile) != 1:
            message = "Sum of wholesaler profile must equal 1"
            logging.error(message)
            raise Exception(message)

        self.wholesalerProfileOption = wholesalerProfileOption
        if self.wholesalerProfileOption != "Optimize":
            if self.wholesalerProfileOption != "Default":
                message = "Unknown option chosen for wholesalerProfileOption"
                logging.error(message)
                raise Exception(message)

    def beginningOfDay(self):
        """
        A function for those aspects of the simulation that the firm needs to finish before the actions in the current
        time period (but after all other firms have finished their previous time period). For actions that do not
        rely on other firms, those methods should go in the endOfDay function.

        A supplier will produce an item the same "time period", but proceeding the actions of the customer. When the
        supplier produces the item, it then updates the PO with how much was actually fulfilled. The wholesaler needs
        this information to calculate its historical return. It therefore needs to be done beginning of day, not end
        of day. NOTE: currently the simulation does not add production time (orders are received, produced, and shipped
        the same day). If this changes, the way that historical return is calculated will have to change as well.
        :return:
        """
        # Might be overridden by logSupplyPo
        self.warningLoop(self.poList, "beginningOfDay()")
        for po in self.poList:
            if po.orderTime == self._timePeriod - 1:
                if po.supplier == 2:
                    self._fulfilledBySupplier[self._timeIndex - 1, 0] = po.fulfilledAmt
                    # First if statement is here to avoid a runtime long_scalar warning
                    if self._ordersBySupplier[self._timeIndex - 1, 0] == 0:
                        amount = 0
                    else:
                        amount = self._fulfilledBySupplier[self._timeIndex - 1, 0] / \
                                 self._ordersBySupplier[self._timeIndex - 1, 0]
                    self._returnBySupplier[self._timeIndex - 1, 0] = amount
                elif po.supplier == 3:
                    self._fulfilledBySupplier[self._timeIndex - 1, 1] = po.fulfilledAmt
                    # First if statement is here to avoid a runtime long_scalar warning
                    if self._ordersBySupplier[self._timeIndex - 1, 1] == 0:
                        amount = 0
                    else:
                        amount = self._fulfilledBySupplier[self._timeIndex - 1, 1] / \
                                 self._ordersBySupplier[self._timeIndex - 1, 1]
                    self._returnBySupplier[self._timeIndex - 1, 1] = amount
        # If it is the first timePeriod, initialize
        if self._timePeriod == 0:
            start = self._timeIndex - self._numTimePeriodsForCalc
            end = self._timeIndex
            self._returnBySupplierForCalc = self._returnBySupplier[start:end, :]
        else:
            # Sets the return by supplier array used for calculations for the previous timePeriod
            self.setReturnBySupplierForCalc(self._timeIndex - 1)

    def calculateCovariance(self, timePeriods):
        """
        A method that will calculate the covariance for the historical return

        :param int timePeriods:
            The number of timePeriods to use to calculate the covariance
        :return:
        """

        # return array needs to be of the form row = variables, column = observation
        start = self._timePeriod - timePeriods - 1
        end = self._timePeriod - 1
        histReturn = self._historicalPortfolio[:, start: end]
        cov = np.cov(histReturn)
        return cov

    def calculateWipOrder(self, amount):
        """
        A method that will take the total amount of WIP required and splits it into an order amount for each supplier

        :param int amount:
            The amount of WIP inventory desired
        :return:
            A list of order amounts
        """
        amount1 = max(0, int(amount * self.wholesalerProfile[0]))
        amount2 = max(0, int(amount * self.wholesalerProfile[1]))
        orderList = [amount1, amount2]
        return orderList

    def checkCovarianceMatrixError(self):
        """
        Perform same check as is done in Configs.py file. needs to be a separate method because the matrix is updated
        each time period??
        """
        pass

    def chooseSuppliers(self, currentTimePeriod):
        """
        The result of this function will be an update to the supplier portfolio.

        :param int currentTimePeriod:
            Current simulation time period
        :return:
        """
        if self.wholesalerProfileOption == 'Optimize':
            self.solveProgram(currentTimePeriod)
        elif self.wholesalerProfileOption == 'Default':
            pass
        else:
            message = f"Firm {self._id}, t={self._timePeriod}, f=chooseSuppliers(): Unknown value chosen for " \
                      f"wholesalerProfileOption"
            logging.error(message)
            raise Exception(message)
        self._historicalPortfolio[self._timeIndex, 0] = self.wholesalerProfile[0]
        self._historicalPortfolio[self._timeIndex, 1] = self.wholesalerProfile[1]

    def createSupplyOrders(self, orderAmount):
        """
        Overrides the Firm3 function to account for the wholesaler having two sources of supply. Creates PO objects
        representing the focal firms supply orders.

        :param int orderAmount:
            Amount of WIP inventory required.

        :return:
            A list of PO objects
        """
        orderList = self.calculateWipOrder(orderAmount)
        newPoList = []
        self.warningLoop(orderList, "createSupplyOrders()")
        for supplier, amount in zip(self.supplierList, orderList):
            newPo = self.createPo(supplier, amount, self._timePeriod)
            newPoList.append(newPo[0])
        self._ordersBySupplier[self._timePeriod + self._historyTime, 0] = orderList[0]
        self._ordersBySupplier[self._timePeriod + self._historyTime, 1] = orderList[1]
        return newPoList

    def logSupplyPo(self, poNumber, timeIndex):
        """
        Overrides the Firm3 method to account for the wholesaler having two sources of supply. Logs data for each PO.

        :param int poNumber:
            Unique PO id
        :param int timeIndex:
            Current simTime + historicalTime
        :return:
        """
        timeIndex -= self.shipDelay
        self.warningLoop(self.poList, "logSupplyPo()")
        for po in self.poList:
            if poNumber == po.id:
                if po.supplier == 2:
                    # This first if statement is put in to avoid a runtime warning regarding "long scalars"
                    if self._ordersBySupplier[timeIndex, 0] == 0:
                        if po.fulfilledAmt > 0:
                            raise Exception("Fulfilled amount larger than ordered amount")
                        else:
                            self._fulfilledBySupplier[timeIndex, 0] = 0
                            # self._returnBySupplier[timeIndex, 0] = 1
                    else:
                        self._fulfilledBySupplier[timeIndex, 0] = po.fulfilledAmt
                        amount = self._fulfilledBySupplier[timeIndex, 0] / self._ordersBySupplier[timeIndex, 0]
                        self._returnBySupplier[timeIndex, 0] = amount
                elif po.supplier == 3:
                    # This first if statement is put in to avoid a runtime warning regarding "long scalars"
                    if self._ordersBySupplier[timeIndex, 1] == 0:
                        if po.fulfilledAmt > 0:
                            raise Exception("Fulfilled amount larger than ordered amount")
                        else:
                            self._fulfilledBySupplier[timeIndex, 1] = 0
                            # self._returnBySupplier[timeIndex, 1] = 1
                    else:
                        self._fulfilledBySupplier[timeIndex, 1] = po.fulfilledAmt
                        amount = self._fulfilledBySupplier[timeIndex, 1] / self._ordersBySupplier[timeIndex, 1]
                        self._returnBySupplier[timeIndex, 1] = amount
                else:
                    raise Exception("Error in logging supply PO")

    def setHistoricalReturnArray(self, historicalWeeklyDemand):
        """
        Overrides the Firm3 function to account for the wholesaler having two sources of supply. Establishes historical
        data that is used in the simulation to optimize the portfolio.

        :param int historicalWeeklyDemand:
            The value that will be used for each week's historical demand.
        :return:
        """
        for time in range(self._historyTime):
            amount1 = max(0, int(historicalWeeklyDemand * self.wholesalerProfile[0]))
            amount2 = max(0, int(historicalWeeklyDemand * self.wholesalerProfile[1]))
            self._ordersBySupplier[time, 0] = amount1
            self._ordersBySupplier[time, 1] = amount2
            self._fulfilledBySupplier[time, 0] = amount1
            self._fulfilledBySupplier[time, 1] = amount1
            self._returnBySupplier[time, 0] = 1
            self._returnBySupplier[time, 1] = 1
            self._historicalPortfolio[time, 0] = self.wholesalerProfile[0]
            self._historicalPortfolio[time, 1] = self.wholesalerProfile[1]

    def setPortfolio(self, profile):
        """
        Sets the instance variable wholesalerProfile to the given value.

        :param list profile:
            A list object where each element is the percentage of supply to be purchased from a given supplier
        :return:
        """
        if np.sum(self.wholesalerProfile) > 1.01:
            message = f"Firm {self._id}, t={self._timePeriod}, f=setPortfolio(): Portfolio sum cannot be greater than 1"
            logging.error(message)
            raise Exception(message)
        self.wholesalerProfile = profile

    def setReturnBySupplierForCalc(self, timeIndex):
        """
        Checks to see if the array should be updated. It will not be updated for a specific firm if there was no order
        placed for that firm in the time period.
        Performed at the beginning of the timePeriod specified as a passed parameter for use in the next timePeriod.
        Checks historicalReturnRuleOption:
        Option 1: Treated as a 0
        Option 2: Treated as a 1
        Option 3 (Default): To avoid such a decision, choosing this option will mean that the wholesaler uses the last n
            timeperiods where an order was sent to that supplier. i.e., It will ignore time periods where order was 0.

        :param int timeIndex:
            An integer value representing the time index for which to calculate the return array.
        :return:
        """
        # Default values for start and end
        start = timeIndex - self._numTimePeriodsForCalc + 1
        end = timeIndex + 1
        if self._historicalReturnRuleOption != 3:  # The easiest cases to handle
            # Only change the value if no order was placed this timeperiod
            if self._ordersBySupplier[timeIndex, 0] == 0:
                if self._historicalReturnRuleOption == 1:
                    self._returnBySupplier[timeIndex, 0] = 0
                elif self._historicalReturnRuleOption == 2:
                    self._returnBySupplier[timeIndex, 0] = 1
            # Only change the value if no order was placed this timeperiod
            if self._ordersBySupplier[timeIndex, 1] == 0:
                if self._historicalReturnRuleOption == 1:
                    self._returnBySupplier[timeIndex, 1] = 0
                elif self._historicalReturnRuleOption == 2:
                    self._returnBySupplier[timeIndex, 1] = 1
            self._returnBySupplierForCalc = self._returnBySupplier[start:end, :]
        # If orders were placed to both suppliers no calculation is needed
        elif self._ordersBySupplier[timeIndex, 0] != 0 and self._ordersBySupplier[timeIndex, 1] != 0:
            self._returnBySupplierForCalc = self._returnBySupplier[start:end, :]
        else:
            if self._ordersBySupplier[timeIndex, 0] == 0:
                # The returnBySupplierForCalc does not change for the first firm
                tempArray1 = self._returnBySupplierForCalc[:, 0]
            else:
                # Update returnBySupplierForCalc with the current timePeriod's return
                tempArray1 = np.append([self._returnBySupplierForCalc[1:, 0]],  # Everything but the first item
                                       [self._returnBySupplier[timeIndex, 0]])
            if self._ordersBySupplier[timeIndex, 1] == 0:
                # The returnBySupplierForCalc does not change for the second firm
                tempArray2 = self._returnBySupplierForCalc[:, 1]
            else:
                # Update returnBySupplierForCalc with the current timePeriod's return
                tempArray2 = np.append([self._returnBySupplierForCalc[1:, 1]],  # Everything but the first item
                                       [self._returnBySupplier[timeIndex, 1]])
            tempArray1 = tempArray1.T
            self._returnBySupplierForCalc = np.stack((tempArray1, tempArray2), axis=-1)

    def solveOptimize(self, currentTimePeriod):
        """
        Solves an optimization problem using the Markowitz portfolio allocation model.
        (H Markowitz. Portfolio Selection: Efficient Diversification of Investment. New York: John Wiley & Sons, 1959.)
        The model is:
        minimize:    W^T*C*W
        subject to: W^T*1(vector) = 1
                    W^T*E = Mu

        Where W is the vector of decision weights, C is the covariance matrix of returns, E is a vector of expected
        returns, and Mu is a measure of risk tolerance. Return is defined for our purposes to be:
         number of WIP received / number of WIP ordered (by supplier).
        Expected return is defined to be the historical average of return, for t-time periods.

        :param currentTimePeriod:
            Current simulation time period.
        :return:
        """
        if currentTimePeriod + self._historyTime != self._timeIndex:
            raise Exception("Time index is off")
        # Grab (numTimePeriodsForCalc) time periods worth of returns
        # startTime = self._timeIndex - self._numTimePeriodsForCalc - 1
        # returnBySupplier = np.transpose(self._returnBySupplier)
        # df = pd.DataFrame(returnBySupplier[:, startTime:self._timeIndex - 1])
        returnBySupplier = np.transpose(self._returnBySupplierForCalc)
        df = pd.DataFrame(returnBySupplier)
        # Calculate expected return
        E = np.array(df.mean(axis=1)).reshape(-1, 1)  # Axis=1 averages across the rows, -1 sets unspecified
        W = np.ones((E.shape[0], 1)) * (1.0 / E.shape[0])  # W sums to 1

        def optimize(func, W, expReturn, covariance, riskProfile):
            optBounds = Bounds(0, 1)  # Weights have to be between 0 and 1
            optConstraints = ({'type': 'eq',
                               'fun': lambda W: 1.0 - np.sum(W)},
                              {'type': 'eq',
                               'fun': lambda W: riskProfile - W.T @ expReturn})
            optWeights = minimize(func, W,
                                  args=(expReturn, covariance),
                                  method='SLSQP',
                                  bounds=optBounds,
                                  constraints=optConstraints)
            return optWeights['x']

        def returnRisk(W, expReturn, covariance):
            return - ((W.T @ expReturn) / (W.T @ covariance @ W) ** 0.5)

        x = optimize(returnRisk, W, E, self.covarianceMatrix, self.riskTolerance)
        for weight in x:
            weight = round(weight, 4)
        self.setPortfolio(x)

    def solveProgram(self, currentTimePeriod):
        """
        Ensures that the portfolio cannot be changed until after historical time is complete.

        :param currentTimePeriod:
            Current simulation time period
        :return:
        """
        # Not allowed to change portfolio until after historical time
        if currentTimePeriod >= 0:
            self.solveOptimize(currentTimePeriod)


class Manufacturer(Firm):
    """
    A subclass that inherits from Firm3, representing a Manufacturer (level 3 in supply chain).

    Instance Methods (Which override Firm3 method)
    ---------
    calculateHistoricalDemand(demandMu, wholesalerProfile):
         Calculates the historical demand, taking into account the chosen value for wholesalerProfile
    """

    def __init__(self,
                 alpha,
                 runTime,
                 historyTime,
                 shipDelay,
                 idNum,
                 customerList=None,
                 customerPoList=None,
                 fgInventory=0,
                 supplierList=None,
                 poList=None,
                 wipInventory=0,
                 ):
        super().__init__(alpha,
                         runTime,
                         historyTime,
                         shipDelay,
                         idNum,
                         customerList,
                         customerPoList,
                         fgInventory,
                         supplierList,
                         poList,
                         wipInventory
                         )

    def calculateHistoricalDemand(self, demandMu, wholesalerProfile=None):
        """
        Calculates the historical demand, taking into account the chosen value for wholesalerProfile

        :param int demandMu:
            Value representing the average end consumer demand
        :param list wholesalerProfile:
            A list object where each element is the percentage of supply to be purchased from a given supplier
        :return:
        """
        if wholesalerProfile is None:
            message = f"Firm {self._id}, t={self._timePeriod}, f=calculateHistoricalDemand(): Wholesaler profile not " \
                      f"initialized"
            logging.error(message)
            raise Exception(message)
        if self.id == 2:  # First index of profile matches with firm 2
            amount = max(0, int(demandMu * wholesalerProfile[0]))
        elif self.id == 3:  # Second index of profile matches with firm 3
            amount = max(0, int(demandMu * wholesalerProfile[1]))
        else:
            message = f"Firm {self._id}, t={self._timePeriod}, f=calculateHistoricalDemand(): Error in assigning " \
                      f"historical demand"
            logging.error(message)
            raise Exception(message)
        return amount


class RawMaterials(Firm):
    """
    A subclass that inherits from Firm3, representing a Manufacturer (level 3 in supply chain).

    Instance Methods (Which override Firm3 method)
    ---------
    createSupplyOrders(orderAmount, currentTimePeriod):
        Overrides the Firm3 function to account for the raw materials producer not having a 'supplier'. The process
        of extracting raw materials is implemented in the simulation as an order from supplier '-99'.
        Creates PO objects representing the focal firms supply orders.
    """

    def __init__(self,
                 alpha,
                 runTime,
                 historyTime,
                 shipDelay,
                 idNum,
                 customerList=None,
                 customerPoList=None,
                 fgInventory=0,
                 supplierList=None,
                 poList=None,
                 wipInventory=0
                 ):
        super().__init__(alpha,
                         runTime,
                         historyTime,
                         shipDelay,
                         idNum,
                         customerList,
                         customerPoList,
                         fgInventory,
                         supplierList,
                         poList,
                         wipInventory
                         )

    def createSupplyOrders(self, orderAmount):
        """
        Overrides the Firm3 function to account for the raw materials producer not having a 'supplier'. The process
        of extracting raw materials is implemented in the simulation as an order from supplier '-99'.
        Creates PO objects representing the focal firms supply orders.

        :param int orderAmount:
            Amount of WIP required
        :return:
            A list of PO objects
        """
        orderAmt = int(orderAmount / len(self.supplierList))
        newPoList = []
        newPo = self.createPo(self.supplierList[0], orderAmt, self._timePeriod)
        newPo[0].updatePO({"fulfilledAmt": orderAmt,
                           "fulfilledTime": self._timePeriod,
                           "arrivalTime": self._timePeriod + self._shipDelay})
        newPo[0].supplierClosed = True
        newPoList.append(newPo[0])
        logging.debug(f"Firm {self._id}, t={self._timePeriod}, f=createSupplyOrders(): Created a new supply order "
                      f"of amount {orderAmount}.")
        return newPoList