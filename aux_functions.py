import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
import pickle
from Auction import random_bidder, run_auction


# extra function to train and test the fitness of GPMW prediction
def func_test(T_train, T_test):
    Q = 30
    N = 5
    c_limit = 100
    d_limit = 50
    K = 10
    bidders = []
    for i in range(N):
        bidders.append(random_bidder(c_limit, d_limit, K, Q))

    # train data
    game_data = run_auction(T_train, bidders, Q, False)
    # input data train
    bids_p1 = [bid[0] for bid in game_data.bids]
    input_data_p1 = []
    for i in range(T_train):
        input_data_p1.append(
            list(game_data.allocations[i]) + [game_data.marginal_prices[i], bids_p1[i][0], bids_p1[i][1]])
    # output data train
    payoff_p1 = [pay[0] for pay in game_data.payoffs]
    gpr = GaussianProcessRegressor(kernel=RBF(), n_restarts_optimizer=10, alpha=0.001 ** 2)
    gpr.fit(np.array(input_data_p1), np.array(payoff_p1))
    # test data
    game_data = run_auction(T_test, bidders, Q, False)
    # input data test
    bids_p1 = [bid[0] for bid in game_data.bids]
    input_data_p1 = []
    for i in range(T_test):
        input_data_p1.append(
            list(game_data.allocations[i]) + [game_data.marginal_prices[i], bids_p1[i][0], bids_p1[i][1]])
    # output data test
    payoff_p1 = [pay[0] for pay in game_data.payoffs]
    mean, std = gpr.predict(input_data_p1, return_std=True)

    # best fit score = 1
    print(r2_score(payoff_p1, mean))
    # plot
    plt.plot(range(T_test), payoff_p1)
    p = plt.plot(range(T_test), mean)
    plt.fill_between(range(T_test), mean - std,
                     mean + std, alpha=0.1, color=p[0].get_color())
    plt.legend(['real payoff', 'predicted payoff'])
    plt.xlabel('Round')
    plt.ylabel('Payoff')
    plt.show()


# extra function to compare theoretical payoff upper bound to practical occurred payoffs
def plot_payoff_upper_bound():
    with open('result.pckl', 'rb') as file:
        T = pickle.load(file)
        c_limit = pickle.load(file)
        d_limit = pickle.load(file)
        Q = pickle.load(file)
        types = pickle.load(file)
        game_data_profile = pickle.load(file)
    for i, typ in enumerate(types):
        data = np.array(
            [[game_data_profile[i][d].payoffs[t][-1] for t in range(T)] for d in range(len(game_data_profile[i]))])
        max_payoff = np.max(data, 0)
        min_payoff = np.min(data, 0)
        plt.plot(range(T), max_payoff, label=f'max payoff {typ}')
        plt.plot(range(T), min_payoff, label=f'min payoff {typ}')
    plt.axhline(y=(c_limit * Q + d_limit) * Q)
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('payoff')
    plt.show()

