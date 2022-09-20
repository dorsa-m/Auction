import numpy as np
import cvxpy as cp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from matplotlib import pyplot as plt


class auction_data:
    def __init__(self):
        self.bids = []
        self.allocations = []
        self.payments = []
        self.marginal_prices = []
        self.payoffs = []
        self.regrets = []


class Bidder:
    def __init__(self, c_limit, d_limit, K):
        self.learning_rate = 1
        self.K = K
        c_list = c_limit * np.random.sample(size=K)
        d_list = d_limit * np.random.sample(size=K)
        self.action_set = list(zip(c_list, d_list))
        # self.action_set = list(zip(np.random.randint(c_limit, size=K), np.random.randint(d_limit, size=K)))
        ratio_c = (c_list.min() / (2 * np.mean(c_list)))
        ratio_d = (d_list.min() / (2 * np.mean(d_list)))
        cost_ratio = min(ratio_c, ratio_d)
        self.cost = (np.mean(c_list) * cost_ratio, np.mean(d_list) * cost_ratio)
        self.weights = np.ones(K)
        self.history_payoff_profile = []
        self.history_action = []
        self.history_payoff = []
        self.cum_each_action = [0] * K
        self.played_action = None

    def update_weights(self, payoffs):
        payoffs = normalize(payoffs, payoffs.min(), payoffs.max())
        losses = np.ones(self.K) - np.array(payoffs)
        self.weights = np.multiply(self.weights, np.exp(np.multiply(self.learning_rate, -losses)))
        self.weights = self.weights / np.sum(self.weights)

    def choose_action(self):
        mixed_strategies = self.weights / np.sum(self.weights)
        choice = np.random.choice(len(self.action_set), p=mixed_strategies)
        return self.action_set[choice], choice


def normalize_util(payoffs, min_payoff, max_payoff):
    if min_payoff == max_payoff:
        return payoffs
    payoff_range = max_payoff - min_payoff
    payoffs = np.maximum(payoffs, min_payoff)
    payoffs = np.minimum(payoffs, max_payoff)
    payoffs_scaled = (payoffs - min_payoff) / payoff_range
    return payoffs_scaled


normalize = np.vectorize(normalize_util)


class Hedge_bidder(Bidder):
    def __init__(self, c_limit, d_limit, K):
        super().__init__(c_limit, d_limit, K)
        self.type = 'Hedge'


class random_bidder(Bidder):
    def __init__(self, c_limit, d_limit, K):
        super().__init__(c_limit, d_limit, K)
        self.type = 'random'


class EXP3_bidder(Bidder):
    def __init__(self, c_limit, d_limit, K):
        super().__init__(c_limit, d_limit, K)
        self.type = 'EXP3'
        self.gamma = 0.9

    def update_weights(self, played_a, payoff):
        payoffs = np.zeros(self.K)
        prob_played_action = (1 - self.gamma) * (self.weights[played_a] / np.sum(self.weights)) + (self.gamma / self.K)
        payoffs[played_a] = payoff / prob_played_action
        super().update_weights(payoffs)


class GPMW_bidder(Bidder):
    def __init__(self, c_limit, d_limit, K):
        super().__init__(c_limit, d_limit, K)
        self.type = 'GPMW'
        self.sigma = 0.001
        self.gpr = GaussianProcessRegressor(kernel=RBF(), n_restarts_optimizer=10, alpha=self.sigma ** 2)
        self.input_history = []
        self.beta = 0.1

    def update_weights(self, alloc_profile):
        self.input_history.append(list(alloc_profile) + [self.played_action[0], self.played_action[1]])
        self.gpr.fit(np.array(self.input_history), np.array(self.history_payoff))
        input_predict = []
        for i in range(self.K):
            input_predict.append(list(alloc_profile) + [self.action_set[i][0], self.action_set[i][1]])
        mean, std = self.gpr.predict(input_predict, return_std=True)
        payoffs = mean + self.beta * std
        super().update_weights(payoffs)


def optimize_alloc(bids, Q):
    C = np.array([param[0] for param in bids])
    C = np.diag(C)
    D = np.array([param[1] for param in bids])
    n = len(bids)
    A = np.ones(n).T
    G = - np.eye(n)
    h = np.zeros(n)

    # non-negativity doesn't strictly hold
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.quad_form(x, C) + D.T @ x),
                      [G @ x <= h, A @ x == Q])
    prob.solve()
    allocs = x.value
    for i in range(len(allocs)):
        if allocs[i] < 10 ** (-5):
            allocs[i] = 0

    # only for quadratic case
    sample_winner = np.argmax(allocs)
    marginal_price = bids[sample_winner][0] * max(allocs) + bids[sample_winner][1]
    payments = marginal_price * allocs

    return allocs, marginal_price, payments


def run_auction(T, bidders, Q):
    game_data = auction_data()
    for t in range(T):
        bids = []
        for bidder in bidders:
            action, ind = bidder.choose_action()
            bidder.played_action = action
            bidder.history_action.append(ind)
            bids.append(action)
        x, marginal_price, payments = optimize_alloc(bids, Q)

        # update actual loss profile and regret(not pseudo)
        regrets = []
        for i, bidder in enumerate(bidders):
            bidder.history_payoff.append(payments[i] - (bidder.cost[0] * x[i] + bidder.cost[1]) * x[i])
            payoffs_each_action = []
            for j, action in enumerate(bidder.action_set):
                tmp_bids = bids.copy()
                tmp_bids[i] = action
                x_tmp, marginal_price_tmp, payments_tmp = optimize_alloc(tmp_bids, Q)
                payoff_action = payments_tmp[i] - (bidder.cost[0] * x_tmp[i] * + bidder.cost[1]) * x_tmp[i]
                payoffs_each_action.append(payoff_action)
                bidder.cum_each_action[j] += payoff_action
            bidder.history_payoff_profile.append(np.array(payoffs_each_action))
            regrets.append((max(bidder.cum_each_action) - sum([bidder.history_payoff[l] for l in range(t)])) / (t + 1))

        # update weights
        payoffs = []
        for i, bidder in enumerate(bidders):
            payoff = payments[i] - bidder.cost[0] * x[i] + bidder.cost[1]
            payoffs.append(payoff)
            if bidder.type == 'Hedge':
                # first start with real regret
                bidder.update_weights(bidder.history_payoff_profile[t])
            if bidder.type == 'EXP3':
                bidder.update_weights(bidder.history_action[t], bidder.history_payoff[t])
            if bidder.type == 'GPMW':
                bidder.update_weights(x)
        game_data.payoffs.append(np.array(payoffs))
        # store data
        game_data.bids.append(bids)
        game_data.allocations.append(x)
        game_data.payments.append(payments)
        game_data.marginal_prices.append(marginal_price)
        game_data.regrets.append(regrets)

    return game_data


def simulate(T_train, T_test):
    Q = 30
    N = 5
    c_limit = 10
    d_limit = 5
    K = 6

    bidders = []
    for i in range(N):
        # bidders.append(random_bidder(c_limit, d_limit, K))
        if i == 0:
            bidders.append(Hedge_bidder(c_limit, d_limit, K))
        if i == 1:
            bidders.append(EXP3_bidder(c_limit, d_limit, K))
        if i == 2:
            bidders.append(GPMW_bidder(c_limit, d_limit, K))
        else:
            bidders.append(random_bidder(c_limit, d_limit, K))

    # train data
    game_data = run_auction(T_train, bidders, Q)
    plt.plot(range(T_train), [game_data.regrets[t][0] for t in range(T_train)])
    plt.plot(range(T_train), [game_data.regrets[t][1] for t in range(T_train)])
    plt.plot(range(T_train), [game_data.regrets[t][2] for t in range(T_train)])
    plt.plot(range(T_train), [game_data.regrets[t][3] for t in range(T_train)])
    plt.legend(['Hedge', 'EXP3','GPMW', 'Random'])
    plt.show()

    # Assumption checking
    # input data
    # bids_p1 = [bid[0] for bid in game_data.bids]
    # input_data_p1 = []
    # for i in range(T_train):
    #     input_data_p1.append(list(game_data.allocations[i]) + [bids_p1[i][0], bids_p1[i][1]])
    # # output data
    # payment_p1 = [pay[0] for pay in game_data.payments]
    # gpr = GaussianProcessRegressor(kernel=RBF(), n_restarts_optimizer=10, alpha=0.001 ** 2)
    # gpr.fit(np.array(input_data_p1), np.array(payment_p1))
    #
    # # test data
    # game_data = run_auction(T_test, bidders, Q)
    # # input data
    # bids_p1 = [bid[0] for bid in game_data.bids]
    # input_data_p1 = []
    # for i in range(T_test):
    #     input_data_p1.append(list(game_data.allocations[i]) + [bids_p1[i][0], bids_p1[i][1]])
    # # output data
    # payment_p1 = [pay[0] for pay in game_data.payments]
    # mean, std = gpr.predict(input_data_p1, return_std=True)
    #
    # # plot
    # plt.plot(range(T_test), payment_p1)
    # p = plt.plot(range(T_test), mean)
    # plt.fill_between(range(T_test), mean - std,
    #                  mean + std, alpha=0.1, color=p[0].get_color())
    # plt.legend(['real payments', 'predicted payments'])
    # plt.show()
    #


simulate(100, 200)
