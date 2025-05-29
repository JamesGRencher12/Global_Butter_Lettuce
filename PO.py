import itertools
import logging


class PO:
    """
    A class to represent a purchase order.


    Class Attributes
    ---------
    idIter: iterator
        Creates a new iterator that will be used to create a unique Identifier for the firm.
        Will increment each time a new object is created.


    Instance Attributes
    ---------
    id: int
        Purchase Order Number
    customer: int
        Unique firm identifier for the customer (PO originator)
    supplier: int
        Unique firm identifier for the supplier (PO fulfiller)
    orderAmt: int
        Quantity of product ordered
    orderTime: int
        Time period the order was placed
    fulfilledAmt: int
        Quantity of product that was produced and  shipped
    fulfilledTime: int
        Time period the order was produced and  shipped
    arrivalTime: int
        Time period the order will arrive (simulation assumes no delays during shipment)
    closed: bool
        The PO will become "closed" when both customerClosed and supplierClosed are true
    customerClosed: bool
        The customer sets this to True when they have received the inventory
    supplierClosed: bool
        The supplier sets this to True when they have produced the inventory


    Instance Methods
    ---------
    closePO():
        Closes the Purchase Order. Checks to see that both the supplier and customer have closed the PO.
        That is the only way it can be closed.

    updatePO(params):
        Takes in a variable amount of parameters and updates the PO
    """
    idIter = itertools.count()

    @classmethod
    def createEndConsumerPO(cls):
        """
        Considering using class methods as factories for the different types of PO's
        :return:
        """
        pass

    def __init__(self,
                 customer,
                 supplier,
                 orderAmt,
                 orderTime,
                 closed=False, ):
        """
        Constructs all the necessary attributes for the PO object

        :param int customer:
            Unique firm identifier for the customer (PO originator)
        :param int supplier:
            Unique firm identifier for the supplier (PO fulfiller)
        :param int orderAmt:
            Quantity of product ordered
        :param int orderTime:
            Time period the order was placed
        :param bool closed:
            The PO will become "closed" when it has been completely fulfilled
        """
        # Will create a new unique PO number
        self._id = next(self.idIter)
        self._customer = customer
        self._supplier = supplier
        self._orderAmt = orderAmt
        self._orderTime = orderTime
        # These will update when the supplier fulfills the order
        self._fulfilledAmt = -99
        self._fulfilledTime = -99
        self._arrivalTime = -99
        # This changes to True when the order has been completely fulfilled
        self._closed = closed
        self._customerClosed = False
        self._supplierClosed = False
        logging.debug(f"Created New PO# {self._id}, Customer#: {self.customer}, Supplier#: {self.supplier}, "
                     f"Order Amount: {self.orderAmt}, Order Time: {self.orderTime}")

    # Getters
    @property
    def id(self):
        return self._id

    @property
    def customer(self):
        return self._customer

    @property
    def supplier(self):
        return self._supplier

    @property
    def orderAmt(self):
        return self._orderAmt

    @property
    def orderTime(self):
        return self._orderTime

    @property
    def fulfilledAmt(self):
        return self._fulfilledAmt

    @property
    def fulfilledTime(self):
        return self._fulfilledTime

    @property
    def arrivalTime(self):
        return self._arrivalTime

    @property
    def closed(self):
        return self._closed

    @property
    def supplierClosed(self):
        return self._supplierClosed

    @supplierClosed.setter
    def supplierClosed(self, value):
        self._supplierClosed = value

    @property
    def customerClosed(self):
        return self._customerClosed

    @customerClosed.setter
    def customerClosed(self, value):
        self._customerClosed = value

    def closePO(self, time):
        """
        Closes the Purchase Order

        Checks to see that both the supplier and customer have closed the PO. That is the only way it can be closed.

        :param int time:
            Current simulation time period
        :return: Bool value whether the PO was successfully closed
        """
        if self._supplierClosed and self._customerClosed:
            self._closed = True
            logging.debug("Attempt to close PO# %s succeeded", self._id)
            return True
        elif not self._supplierClosed:
            logging.debug("Attempted to close PO# %s but the supplier had not closed yet", self._id)
            return False
        else:
            logging.debug("Attempted to close PO# %s but the customer had not closed yet", self._id)
            return False

    def updatePO(self, params):
        """
        Updates the Purchase Order

        Takes in a variable amount of parameters and updates the po

        Parameters
        ----------
        fulfilledAmt: int (optional)
            Appends to the fulfilledAmt list item
        fulfilledTime: int (optional)
            Appends to the fulfilledTime list item
        arrivalTime: int (optional)
            Appends to the arrivalTime list item

        :return: None
        """
        fulfilledAmt = params.get('fulfilledAmt')
        if fulfilledAmt is not None:
            self._fulfilledAmt = fulfilledAmt
            logging.debug('Parameter fulfilledAmt changed to value %s for PO# %s', fulfilledAmt, self._id)
        fulfilledTime = params.get('fulfilledTime')
        if fulfilledTime is not None:
            self._fulfilledTime = fulfilledTime
            logging.debug('Parameter fulfilledTime changed to value %s for PO# %s', fulfilledTime, self._id)
        arrivalTime = params.get('arrivalTime')
        if arrivalTime is not None:
            self._arrivalTime = arrivalTime
            logging.debug('Parameter arrivalTime changed to value %s for PO# %s', arrivalTime, self._id)