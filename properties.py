class Constants:
    def __init__(self):
        self.__mu_water = 5.531e-6  # m^2 / s
        self.__mu_oil = 2.9e-4  # m^2 / s
        self.__B_w = 1.  # relative
        self.__B_o = 1.  # relative
        self.__c_o = 15.8e-10   # Pa^-1 https://www.sciencedirect.com/topics/engineering/oil-compressibility#:~:text=Oil%20compressibility%20(also%20called%20isothermal,10−6%20psi−1.
        self.__c_w = 15.8e-10  # Pa^-1
        self.__c_r = 1e-6 / 6894  # Pa^-1 https://www.sciencedirect.com/topics/engineering/formation-compressibility
        self.__k_avg = 1e-1 * 1.987e-13  # 1 darcy to m^2
        self.__dt = 0.1  # s
        self.__dx = 1.  # m
        self.__dy = 1.  # m
        self.__d_avg = 5.  # m
        self.__p_0 = 4e4 * 6894  # psi to Pa
        self.__r_well = 0.1  # m
        self.__delta_p = 20 * 6894 # psi to Pa

    def dx(self):
        return float(self.__dx)

    def dy(self):
        return self.__dy

    def depth_avg(self):
        return self.__d_avg

    def dt(self):
        return self.__dt

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

    def k_r_liq(self, x):
        return x

    def k_avg(self):
        return self.__k_avg

    def c_r(self):
        return self.__c_r

    def c_w(self):
        return self.__c_w

    def c_o(self):
        return self.__c_o

    def p_0(self):
        return self.__p_0

    def r_well(self):
        return self.__r_well

    def delta_p(self):
        return self.__delta_p
