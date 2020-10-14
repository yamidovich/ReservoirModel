class Constants:
    def __init__(self):
        self.__mu_water = 5.531e-7  # m^2 / s
        self.__mu_oil = 2.86e-6  # m^2 / s
        self.__B_w = 1  # relative
        self.__B_o = 1  # relative
        self.__c_o = 45.8e-11   # Pa^-1
        self.__c_w = 45.8e-11  # Pa^-1
        self.__c_r = 1e-9  # Pa^-1
        self.__k_r_o = 1  # relative
        self.__k_r_w = 1  # relative
        self.__k_avg = 1 * 1.987e-13  # 1 darcy as m^2
        self.dt = 10  # s
        self.dx = 100  # m
        self.dy = 100  # m

    def mu_water(self):
        return self.__mu_water

    def mu_oil(self):
        return self.__mu_oil

    def c_t(self):
        return self.__c_r + self.__c_o + self.__c_w

    def b_w(self):
        return self.__B_w

    def b_o(self):
        return self.__B_o

    def k_r_o(self):
        return self.__k_r_o

    def k_r_w(self):
        return self.__k_r_w

    def c_r(self):
        return self.__c_r

    def c_w(self):
        return self.__c_w

    def c_o(self):
        return self.__c_o
