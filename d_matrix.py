from math import ceil, floor
import utils as u

class DMatrix:
    """
    class for matrix of depth. we got inter block values, but what do we have on boundaries...
    so we have to call [1, 4.5] to get proper inter block value
    """

    def __init__(self, d_values):
        self.__d = d_values
        self.shape = d_values.shape

    def __getitem__(self, item):
        """
        returns an inter block value or in-block...
        :param item: [1.5, 3.5], not [1.5][3.5]
        :return:
        """
        # simplest case
        if u.check_if_numerical(self.__d):
            return self.__d
        # major case - tuple
        if (type(item) == tuple) & (len(item) == 2):
            i, j = item

            # bound
            if i < 0:
                return self.__d[0, j]
            if i >= self.__d.shape[0] - 0.6:
                return self.__d[self.__d.shape[0] - 1, j]
            if j < 0:
                return self.__d[i, 0]
            if j >= self.__d.shape[1] - 0.6:
                return self.__d[i, self.__d.shape[1] - 1]

            if u.check_int(i) & u.check_half(j):
                out = self.__d[floor(i), floor(j)] + self.__d[floor(i), ceil(j)]
                out *= 0.5
                return out

            if u.check_half(i) & u.check_int(j):
                out = self.__d[floor(i), floor(j)] + self.__d[ceil(i), floor(j)]
                out *= 0.5
                return out
            # if for some reason not boundary condition
            if u.check_int(i) & u.check_int(j):
                return self.__d[item]

        elif type(item) == int:
            return self.__d[item]