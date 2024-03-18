import numpy as np
import matplotlib.pyplot as plt
import math


def Function(x, y):
    if FUNCTION_TYPE == 'duff_delta':
        res = np.array([y[1], 10*y[1]-(y[0] + 0.1*(y[0]**3))])
    if FUNCTION_TYPE == 'duff_pos':
        res = np.array([y[1], -(y[0] + 0.1*(y[0]**3))])
    if FUNCTION_TYPE == 'duff_neg':
        res = np.array([y[1], -(y[0] - 0.1*(y[0]**3))])
    if FUNCTION_TYPE == 'y-y':
        res = np.array([y[1], -y[0]])
    if FUNCTION_TYPE == 'yy':
        res = np.array([y[1], y[0]])
    if FUNCTION_TYPE == 'y':
        res = y*np.ones(y.shape)
    if FUNCTION_TYPE == 'x':
        res = x*np.ones(y.shape)
    if FUNCTION_TYPE == 'x^2':
        res = (x**2)*np.ones(y.shape)
    if FUNCTION_TYPE == '1':
        res = np.ones(y.shape)
    return res


def Step_I(x, y, h):
    k1 = h*Function(x, y)
    k2 = h*Function(x + 0.5*h, y + 0.5*k1)
    k3 = h*Function(x + h, y - k1 + 2.*k2)
    res = y + (1./6) * (k1 + 4*k2 + k3)
    return res


def Step_II(x, y, h):
    k1 = h*Function(x, y)
    k2 = h*Function(x + 0.5*h, y + 0.5*k1)
    k3 = h*Function(x + 0.5*h, y + 0.5*k2)
    k4 = h*Function(x + h, y + k3)
    res = y + (1./6) * (k1 + 2*k2 + 2*k3 + k4)
    return res


def Step_III(x, y, h):
    k1 = h*Function(x, y)
    k2 = h*Function(x + (1./3)*h, y + (1./3)*k1)
    k3 = h*Function(x + (1./3)*h, y + (1./6)*(k1 + k2))
    k4 = h*Function(x + 0.5*h, y + (1./8)*k1 + (3./8)*k3)
    k5 = h*Function(x + h, y + 0.5*k1 - (3./2)*k3 + 2*k4)
    E = (1./30)*(2.*k1 - 9.*k3 + 8.*k4 - k5)
    res = y + (1./6) * (k1 + 4*k4 + k5)
    return E, res


def I_method(x_0, x_end, y_0, eps):
    print(10*'----')
    print("I_method is in process: \n")
    leng = x_end - x_0
    func_use = 0
    x = np.array([])
    y = []
    h = x_end - x_0
    x_cur = x_0
    y_cur = y_0
    y_next = y_0
    y_cur_double = y_0
    x = np.append(x, x_cur)
    y.append(y_cur)
    while abs(x_cur - x_end) > eps:
        if (x_cur + h) > x_end:
            h = x_end - x_cur
        ro = 10000
        while abs(ro) > eps*2*h/leng:
            y_next = Step_I(x_cur, y_cur, h)
            func_use += 3
            h_double = h/2
            y_next_double = Step_I(x_cur + h_double, Step_I(x_cur, y_cur_double, h_double), h_double)
            func_use += 6
            ro = np.max((y_next - y_next_double))/(1. - 2**(-2))
            h = h/2
        h *= 2
        x_cur = x_cur + h
        y_cur = y_next
        y_cur_double = y_next
        h *= 2
        x = np.append(x, x_cur)
        y.append(y_cur)
    print("With {} step(s) and {} using of right part\n We have in point {}:\n".format(x.shape[0] - 1, func_use, x[-1]))
    for i, res in enumerate(y[-1]):
        print("y[{}] : {}\n".format(i, res))
    print(10*'----')
    return x, y


def II_method(x_0, x_end, y_0, eps):
    print(10*'----')
    print("II_method is in process: \n")
    leng = x_end - x_0
    func_use = 0
    x = np.array([])
    y = []
    h = x_end - x_0
    x_cur = x_0
    y_cur = y_0
    y_next_dop = y_0
    y_cur_dop = y_0
    x = np.append(x, x_cur)
    y.append(y_cur)
    while abs(x_cur - x_end) > eps:
        if (x_cur + h) > x_end:
            h = x_end - x_cur
        ro = 10000
        while abs(ro) > eps*2*h/leng:
            y_next = Step_I(x_cur, y_cur, h)
            func_use += 3
            y_next_dop = Step_II(x_cur, y_cur_dop, h)
            func_use += 4
            ro = np.max(abs(y_next - y_next_dop))/(1. - 2**(-3))
            h = h/2
        h *= 2
        x_cur = x_cur + h
        y_cur = y_next_dop
        y_cur_dop = y_next_dop
        h *= 2
        x = np.append(x, x_cur)
        y.append(y_cur)
    print("With {} step(s) and {} using of right part\n We have in point {}:\n".format(x.shape[0] - 1, func_use, x[-1]))
    for i, res in enumerate(y[-1]):
        print("y[{}] : {}\n".format(i, res))
    print(10*'----')
    return x, y


def III_method(x_0, x_end, y_0, eps):
    print(10*'----')
    print("III_method is in process: \n")
    leng = x_end - x_0
    func_use = 0
    x = np.array([])
    y = []
    h = x_end - x_0
    x_cur = x_0
    y_cur = y_0
    y_next = y_0
    x = np.append(x, x_cur)
    y.append(y_cur)
    while abs(x_cur - x_end) > eps:
        if (x_cur + h) > x_end:
            h = x_end - x_cur
        E = 10000
        while abs(E) > eps*2*h/leng:
            E, y_next = Step_III(x_cur, y_cur, h)
            func_use += 5
            E = np.max(np.abs(E))
            h = h/2
        h *= 2
        x_cur = x_cur + h
        y_cur = y_next
        h *= 2
        x = np.append(x, x_cur)
        y.append(y_cur)
    print("With {} step(s) and {} using of right part\n We have in point {}:\n".format(x.shape[0] - 1, func_use, x[-1]))
    for i, res in enumerate(y[-1]):
        print("y[{}] : {}\n".format(i, res))
    print(10*'----')
    return x, y


def main():
    x_0 = 0
    y_0 = np.array([1, 1])
    x_end = 1
    eps = 1e-9
    x1, y1 = I_method(x_0, x_end, y_0, eps)
    x2, y2 = II_method(x_0, x_end, y_0, eps)
    x3, y3 = III_method(x_0, x_end, y_0, eps)
    y1 = np.array(y1).T
    y2 = np.array(y2).T
    y3 = np.array(y3).T

    plt.figure()
    for i, res in enumerate(y1):
        plt.plot(x1, res)
        plt.xlabel('x')
        plt.ylabel("y[{}]".format(i))
        plt.title('I Method')
        plt.show()
    if y1.shape[0] > 1:
        plt.plot(y1[0], y1[1])
        plt.xlabel("y[{}]".format(0))
        plt.ylabel("y[{}]".format(1))
        plt.show()

    plt.figure()
    for i, res in enumerate(y2):
        plt.plot(x2, res)
        plt.xlabel('x')
        plt.ylabel("y[{}]".format(i))
        plt.title('II Method')
        plt.show()
    if y2.shape[0] > 1:
        plt.plot(y2[0], y2[1])
        plt.xlabel("y[{}]".format(0))
        plt.ylabel("y[{}]".format(1))
        plt.show()

    plt.figure()
    for i, res in enumerate(y3):
        plt.plot(x3, res)
        plt.xlabel('x')
        plt.ylabel("y[{}]".format(i))
        plt.title('III Method')
        plt.show()
    if y3.shape[0] > 1:
        plt.plot(y3[0], y3[1])
        plt.xlabel("y[{}]".format(0))
        plt.ylabel("y[{}]".format(1))
        plt.show()


FUNCTION_TYPE = 'yy'
main()
