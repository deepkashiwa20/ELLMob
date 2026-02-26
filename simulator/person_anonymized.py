"""Represents a person with train and test routine lists and a name."""
class Person:
    def __init__(self, name):
        self.train_routine_list = None  # list of training routines
        self.test_routine_list = None  # list of testing routines
        self.name = name
        print("Person {} is created".format(self.name))







