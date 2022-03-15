import numpy as np
import matplotlib.pyplot as plt
from kafe2 import XYContainer, Fit, Plot, ContoursProfiler


def xy_data(x_data, y_data, x_err=0, y_err=0):
    '''
    Function to make code more clear. Builds the XYContainer used by kafe2
    :param x_data: x-data for fit
    :param y_data: y-data for fit
    :param x_err: error of x-data for fit
    :param y_err: error of y-data for fit
    :return:
    '''
    data = XYContainer(x_data=x_data, y_data=y_data)
    data.add_error(axis='x', err_val=x_err)
    data.add_error(axis='y', err_val=y_err)
    return data


#########
# 370.a #
#########

def a_model(x, i_0, b, i_1):
    '''
    Fit model for 370.a
    :param x: variable parameter for kafe2
    :param i_0: intensity of light
    :param b: correctional parameter for when phi_0 wasn't calculated correctly
    :param i_1: intensity of not perfectly polarized light
    :return: model function
    '''
    return i_0 * np.cos(np.deg2rad(x) + b) ** 2 + i_1


def a_fit(x, y, x_err, y_err):
    '''
    Fit function to fit data according to 370.a.
    :param x:
    :param y:
    :param x_err:
    :param y_err:
    :return:
    '''
    data = xy_data(x, y, x_err, y_err)

    fit = Fit(data=data, model_function=a_model)
    results = fit.do_fit()
    fit.report()

    i_0 = results['parameter_values']['i_0']
    b = results['parameter_values']['b']
    i_1 = results['parameter_values']['i_1']

    data_range = np.linspace(x[0], x[np.size(x) - 1], np.size(x) * 5)
    y_fit = a_model(data_range, i_0, b, i_1)

    fig, ax = plt.subplots()
    ax.errorbar(x, y, fmt='x', xerr=x_err, yerr=y_err, label='Datenpunkte', color='#004287', capsize=1)
    ax.plot(data_range, y_fit, label='Fit', color='#e94653')

    ax.grid(True)
    ax.set_title("Intensität je nach Drehung der Polarisatoren")
    ax.set_xlabel("φ-φ₀ [°]")
    ax.set_ylabel("Intensität [V]")

    ax.legend()

    fig.savefig(f'370_a.pdf')

    plt.show()


a_data = np.loadtxt('a.txt')
# -90 -90 0.5 0 0.8 9.97 0.01 9.956 0.011 manually removed line (formerly) 20 to avoid double x values
a_angle = a_data[:, 3]
a_angle_err = a_data[:, 4]
a_voltage = a_data[:, 7]
a_voltage_err = a_data[:, 8]

a_fit(a_angle, a_voltage, a_angle_err, a_voltage_err)


#########
# 370.b #
#########


def b_model(x, a, b):
    '''
    Fit model for 370.b
    :param x: the inverse wavelength squared (not just wavelength)
    :param a: constant
    :param b: constant
    :return:
    '''
    return x * b + a


def b_fit(x, y, y_err):
    '''
    Fit function for 370.b
    :param x: 1/wavelength**2
    :param y: angle
    :param y_err: error angle
    :return: fit
    '''
    data = xy_data(x, y, 0, y_err)

    fit = Fit(data=data, model_function=b_model)
    results = fit.do_fit()
    fit.report()

    a = results['parameter_values']['a']
    b = results['parameter_values']['b']

    data_range = np.linspace(x[0], x[np.size(x) - 1], np.size(x) * 5)
    y_fit = b_model(data_range, a, b)

    fig, ax = plt.subplots()
    ax.errorbar(x, y, fmt='x', yerr=y_err, label='Datenpunkte', color='#004287', capsize=1)
    ax.plot(data_range, y_fit, label='Fit', color='#e94653')

    ax.grid(True)
    ax.set_title("Auftragung der Biotschen-Formel")
    ax.set_xlabel("1/λ² [1/nm²]")
    ax.set_ylabel("φ-φ₀ [°]")

    ax.legend()

    fig.savefig(f'370_b.pdf')

    plt.show()


b_angle = np.array([32.5, 20.7, -0.7, -21.6, -41.8, -61.5, -71.9])
b_angle_err = np.full(np.size(b_angle), 0.27)
b_wavelength = np.array([694, 620, 568, 520, 488, 458, 430])

b_fit(1/b_wavelength**2, b_angle, b_angle_err)


#########
# 370.c #
#########


def c_model(x, a, b):
    # DONE: find fit function (probably linear)
    return x*a + b


def c_fit(x, y, y_err):
    '''
    Fit function for 370.c
    :param x: concentration [mol/l]
    :param y: phi-phi_0
    :param y_err: error of y data
    :return: Fit
    '''
    data = xy_data(x, y, 0, y_err)

    fit = Fit(data=data, model_function=c_model)
    results = fit.do_fit()
    fit.report()

    a = results['parameter_values']['a']
    b = results['parameter_values']['b']

    data_range = np.linspace(x[0], x[np.size(x) - 1], np.size(x) * 5)
    y_fit = c_model(data_range, a, b)

    fig, ax = plt.subplots()
    ax.errorbar(x, y, fmt='x', yerr=y_err, label='Datenpunkte', color='#004287', capsize=1)
    ax.plot(data_range, y_fit, label='Fit', color='#e94653')

    ax.grid(True)
    ax.set_title("Drehvermögen in Abhängigkeit der Konzentration")
    ax.set_xlabel("Konzentration [mol/l]")
    ax.set_ylabel("φ-φ₀ [°]")

    ax.legend()

    fig.savefig(f'370_c.pdf')

    plt.show()


c_angle = np.array([29, 35.9, 38.6, 53.9, 66.6])
c_angle_err = np.full(np.size(c_angle), 0.4)
c_concentration = np.array([5, 4, 3, 2, 1])

c_fit(c_concentration, c_angle, c_angle_err)

