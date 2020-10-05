import numpy as np
from matrixes import KMatrix


class Constants:
    def __init__(self, k_values, dy_matrix, dx_matrix):
        self.__mu_water = 1
        self.__mu_oil = 1
        self.__c_t = 1
        self.__c = 1
        self.__k_matrix = KMatrix(k_values, dy_matrix, dx_matrix)

    def mu_water(self):
        return self.__mu_water

    def mu_oil(self):
        return self.__mu_oil

    def c_t(self):
        return self.__c_t

    def c(self):
        return self.__c

    def k(self, item):
        """
        returns inter cell value, or in cell value
        :param item: [1.5, 3] or [1, 1.4] or whatever
        :return:
        """
        return self.__k_matrix[item]