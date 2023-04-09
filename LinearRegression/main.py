import sys
import pandas as pd
import numpy
import math
import matplotlib.pyplot as plt


def dataset(fish):
    reader = pd.read_csv("Fish.csv")
    reader = reader[reader["Species"].str.contains(f"{fish}")]
    x = reader.Length2.values
    y = reader.Weight.values

    return x, y


def compute_cost(x, y, w, b):
    m = x.shape[0]
    total_cost = 0

    for i in range(m):
        fw_b = w*x[i] + b
        err = fw_b - y[i]
        total_cost += err**2
    total_cost = total_cost/(2 * m)
    return total_cost


def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        fw_b = x[i] * w + b
        dj_dw_i = (fw_b - y[i]) * x[i]
        dj_db_i = (fw_b - y[i])

        dj_dw += dj_dw_i / m
        dj_db += dj_db_i / m

    return dj_dw, dj_db


def gradient_descent(x, y, w, b, iteration_count, alpha):

    j_history = []

    for i in range(iteration_count):
        dj_dw, dj_db = compute_gradient(x, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(iteration_count / 10) == 0:
            cost = compute_cost(x, y, w, b)
            j_history.append(cost)
            print(f"Iteration {i:4}: Cost {float(j_history[-1]):8.2f}")

    print("w,b found by gradient descent:", w, b)
    return w, b


if __name__ == '__main__':
    numpy.seterr(all="raise")
    fish_name_list = ["Perch", "Bream", "Roach", "Whitefish", "Parkki", "Pike", "Smelt"]
    while 1:
        fish_name = input(f"Enter a Fish Name \n{fish_name_list}:").title()
        if fish_name in fish_name_list:
            break
        else:
            print("\nEnter a valid name.\n")

    x_train, y_train = dataset(fish_name)

    iterations = int(input("Input iteration count:"))
    alpha = float(input("Input alpha(0.001~ recommended):"))

    try:
        w_final, b_final = gradient_descent(x_train, y_train, 2, 2, iterations, alpha)

    except Exception as exp:
        print(exp, "\nPlease try again")
        input()
        sys.exit()

    else:
        m = x_train.shape[0]
        prediction = numpy.zeros(m)

        for i in range(m):
            prediction[i] = w_final * x_train[i] + b_final

        plt.plot(x_train, prediction, c="b")
        plt.scatter(x_train, y_train, marker="X", c="r")
        plt.title(f"Length vs Weight for {fish_name}")
        plt.ylabel("Weight")
        plt.xlabel("Length")
        plt.show()

    while 1:
        try:
            user_length = float(input("\nEnter a diagonal length(cm) to estimate its weight:"))
        except Exception as exp:
            print(exp, "\nPlease try again.")
        else:
            print(f"\nA {fish_name} that has a {user_length} cm of length is nearly {(w_final * user_length + b_final):.2f} gram.")
