"""
Although representing a game in its normal form is harder to interpret for humans, it is more
convenient for computers. Finding strictly dominant strategies and Nash equilibria are also easier
in normal forms. In this assignment you are asked to write a program to find a mixed Nash
equilibrium of a two player game given in normal form. Your program can output any of the mixed
Nash equilibria (why there exist at least one Nash equilibrium?).

Input Specification
Your program has to read inputs from the standard input. The first line of the input is an integer
n denoting the size of the payoff matrices. After that, the payoff matrix of the first player follows.
The payoff matrix contains n lines each with n numbers where each number is in range [−100, 100].
After that, the payoff matrix of the second player follows which appears as n lines each with n
numbers from range [−100, 100].

Output Specification
Your program has to write the answers into the standard output. The output has to be two lines
each with n numbers. The ith number of the first line should denote the probability of playing ith
row for the first player. The ith number of the second line should denote the probability of playing
ith column for the second player. Writing up to 4 digits after the decimal point suffices for each
number of the output. In the following you can see a few samples. Note that for each sample there
might be multiple correct answers and all of them are acceptable.

Sample Input 1
3
-1 1 1
1 -1 1
-1 1 1
1 1 -1
-1 -1 1
1 -1 1

Sample Output 1
0.5 0.5 0.0
0.5 0.5 0.0
"""

import numpy as np
import pulp
import scipy


def main ():
    # take in the size of the matrix
    n = int(input())

    # initialize NumPy arrays to store the input matrices
    m1 = np.zeros((n, n), dtype=int)  # player 1's matrix
    m2 = np.zeros((n, n), dtype=int)  # player 2's matrix

    # function to validate that input is between -100 and 100
    def validate_input(value):
        return -100 <= value <= 100

    # take in the two matrices and check that input is between -100 and 100
    for i in range(n):
        row_values = list(map(int, input().split()))
        while len(row_values) != n or not all(validate_input(val) for val in row_values):
            print("Invalid input. Please enter", n, "integer values between -100 and 100:")
            row_values = list(map(int, input().split()))
        m1[i, :] = np.array(row_values)

    for i in range(n):
        row_values = list(map(int, input().split()))
        while len(row_values) != n or not all(validate_input(val) for val in row_values):
            print("Invalid input. Please enter", n, "integer values between -100 and 100:")
            row_values = list(map(int, input().split()))
        m2[i, :] = np.array(row_values)

    print(m1)
    print(m2)

    # create vars
    nash = pulp.LpProblem("mixed_eq", pulp.LpMaximize)

    # create decision variables
    x = [pulp.LpVariable(f'x{i}', lowBound=0, upBound=1) for i in range(n)]
    y = [pulp.LpVariable(f'y{j}', lowBound=0, upBound=1) for j in range(n)]

    print(x)
    print(y)

    # objective function
    #obj_p1 = sum(x[i] * sum(x[j] * m1[i][j] for j in range(n)) for i in range(n))
    #obj_p2 = sum(x[j] * sum(y[i] * m2[i][j] for i in range(n)) for j in range(n))
    #ob1 = np.sum(x * np.dot(m1, y))
    #ob2 = np.sum(y * np.dot(m2, x))
    # objective functions
    #ob1 = sum(x[i] * sum(y[j] * m1[i, j] for j in range(n)) for i in range(n))
    #ob2 = sum(y[j] * sum(x[i] * m2[i, j] for i in range(n)) for j in range(n))
    #ob1 = sum(x[i].value() * sum(y[j] * m1[i, j] for j in range(n)) for i in range(n))
    #ob2 = sum(y[j].value() * sum(x[i] * m2[i, j] for i in range(n)) for j in range(n))

    #ob1 = x[0]*(m1[0][0]*y[0] + m1[0][1]*y[1] + m1[0][2]*y[2]) + x[1]*(m1[1][0]*y[0] + m1[1][1]*y[1] + m1[1][2]*y[2]) + x[2]*(m1[2][0]*y[0] + m1[2][1]*y[1] + m1[2][2]*y[2])
    #ob2 = y[0]*(m2[0][0]*x[0] + m2[0][1]*x[1] + m2[0][2]*x[2]) + y[1]*(m2[1][0]*x[0] + m2[1][1]*x[1] + m2[1][2]*x[2]) + y[2]*(m2[2][0]*x[0] + m2[2][1]*x[1] + m2[2][2]*x[2])
    #ob1 = sum(x[i].value() * sum(y[j].value() * m1[i, j] for j in range(n)) for i in range(n))
    #ob2 = sum(y[j].value() * sum(x[i].value() * m2[i, j] for i in range(n)) for j in range(n))

    # add the objective functions to the LP problem
    #nash += ob1, "Player_1_Objective"
    #nash += ob2, "Player_2_Objective"

    # add constraints for valid probability distributions
    #     make sure the sum of the probabilities is 1
    nash += sum(x[i] for i in range(n)) == 1, "Player_1_Probability_Distribution"
    nash += sum(y[j] for j in range(n)) == 1, "Player_2_Probability_Distribution"
    #    make sure the probabilities are non-negative
    for i in range(n):
        nash += x[i] >= 0, f"Player_1_Non_Negative_{i}"
        nash += y[i] >= 0, f"Player_2_Non_Negative_{i}"

    # Solve the LP problem
    nash.solve()

    # Extract and print the mixed strategies
    if pulp.LpStatus[nash.status] == "Optimal":
        mixed_strategy_player_1 = [pulp.value(x[i]) for i in range(n)]
        mixed_strategy_player_2 = [pulp.value(y[j]) for j in range(n)]
        print("Mixed Strategy for Player 1:", mixed_strategy_player_1)
        print("Mixed Strategy for Player 2:", mixed_strategy_player_2)
    else:
        print("No Nash Equilibrium found.")


main()
