import random
import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
import preprocess

class DeepQTrader:
    ACTIONS_COUNT = 0  # number of valid actions. In this case buy, hold and sell for each stock
    INPUT_COUNT = 0 # number of features
    STOCK_COUNT=0 # number of stock
    EXPLORE_STEPS = 10000.  # frames over which to anneal epsilon
    FUTURE_REWARD_DISCOUNT = 0.99  # decay rate of past observations
    INITIAL_RANDOM_ACTION_PROB = 1.0  # starting chance of an action being random
    FINAL_RANDOM_ACTION_PROB = 0.05  # final chance of an action being random
    LENGTH_TRADE_MIN=30
    LENGTH_TRADE_MAX = 120
    LEARN_RATE = 0.1
    MAX_PORTFOLIO_SIZE=20

    def __init__(self, ACTIONS_COUNT,INPUT_COUNT,STOCK_COUNT, NUM_HIDDEN,LEARN_RATE,verbose_logging=False,restore=False,checkpoint_path="model",results_name=""):

        self.ACTIONS_COUNT=ACTIONS_COUNT
        self.INPUT_COUNT = INPUT_COUNT
        self.STOCK_COUNT = STOCK_COUNT
        self.LEARN_RATE = LEARN_RATE

        self.verbose_logging = verbose_logging
        self._checkpoint_path=checkpoint_path
        self._restore=restore
        self._results_name=results_name
        self._session = tf.Session()
        self._input_layer, self._output_layer = DeepQTrader._create_network(self.INPUT_COUNT,self.ACTIONS_COUNT,NUM_HIDDEN)

        self._action = tf.placeholder("float", [None, self.ACTIONS_COUNT])
        self._target = tf.placeholder("float", [None])

        readout_action = tf.reduce_sum(tf.mul(self._output_layer, self._action), reduction_indices=1)

        cost = tf.reduce_mean(tf.square(self._target - readout_action))
        self._train_operation = tf.train.AdamOptimizer(self.LEARN_RATE).minimize(cost)



        self._probability_of_random_action = self.INITIAL_RANDOM_ACTION_PROB


        self._session.run(tf.initialize_all_variables())

        self._saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(self._checkpoint_path)

        if self._restore:
            self._saver.restore(self._session, checkpoint.model_checkpoint_path)
            print("Loaded checkpoints %s" % checkpoint.model_checkpoint_path)

    @staticmethod
    def _create_network(NUM_INPUTS, NUM_OUTPUTS, NUM_HIDDEN):
        # network weights
        feed_forward_weights_3 = None
        feed_forward_bias_3 = None
        if NUM_HIDDEN == 1:
            feed_forward_weights_1 = tf.Variable(
                tf.truncated_normal([NUM_INPUTS, (NUM_INPUTS + NUM_OUTPUTS) / 2], stddev=0.01))
            feed_forward_bias_1 = tf.Variable(tf.constant(0.01, shape=[(NUM_INPUTS + NUM_OUTPUTS) / 2]))
        else:
            feed_forward_weights_1 = tf.Variable(
                tf.truncated_normal([NUM_INPUTS, NUM_INPUTS - (NUM_INPUTS - NUM_OUTPUTS) / 3], stddev=0.01))
            feed_forward_bias_1 = tf.Variable(tf.constant(0.01, shape=[NUM_INPUTS - (NUM_INPUTS - NUM_OUTPUTS) / 3]))

        if NUM_HIDDEN == 1:
            feed_forward_weights_2 = tf.Variable(
                tf.truncated_normal([(NUM_INPUTS + NUM_OUTPUTS) / 2, NUM_OUTPUTS], stddev=0.01))
            feed_forward_bias_2 = tf.Variable(tf.constant(0.01, shape=[NUM_OUTPUTS]))
        else:
            feed_forward_weights_2 = tf.Variable(tf.truncated_normal(
                [NUM_INPUTS - (NUM_INPUTS - NUM_OUTPUTS) / 3, NUM_INPUTS - 2 * (NUM_INPUTS - NUM_OUTPUTS) / 3],
                stddev=0.01))
            feed_forward_bias_2 = tf.Variable(
                tf.constant(0.01, shape=[NUM_INPUTS - 2 * (NUM_INPUTS - NUM_OUTPUTS) / 3]))
            feed_forward_weights_3 = tf.Variable(
                tf.truncated_normal([NUM_INPUTS - 2 * (NUM_INPUTS - NUM_OUTPUTS) / 3, NUM_OUTPUTS], stddev=0.01))
            feed_forward_bias_3 = tf.Variable(tf.constant(0.01, shape=[NUM_OUTPUTS]))

        input_layer = tf.placeholder("float", [None, NUM_INPUTS])

        hidden_activations = tf.nn.relu(
            tf.matmul(input_layer, feed_forward_weights_1) + feed_forward_bias_1)

        if NUM_HIDDEN == 1:
            output_layer = tf.matmul(hidden_activations, feed_forward_weights_2) + feed_forward_bias_2
        else:
            hidden_activations2 = tf.nn.relu(
                tf.matmul(hidden_activations, feed_forward_weights_2) + feed_forward_bias_2)
            output_layer = tf.matmul(hidden_activations2, feed_forward_weights_3) + feed_forward_bias_3

        return input_layer, output_layer

    #Add holds usgin portfolio
    def _add_holds(self, previous_states):
        holds=np.zeros(self.STOCK_COUNT,type='float')
        for p in self.portfolio:
            holds[p[0]]=1.0
        return np.append(previous_states,holds)

    #Update portfolio given action
    def _update_portfolio(self, prices, date, trades_log=None):
        index = int(self.action_index / 3)
        a = self.action_index % 3
        #print "Portfolio Update action {} index {} portofolio size {}".format(a,index,len(self.portfolio))
        if a == 0:

            self.portfolio.append([index, prices.columns[index], prices[prices.columns[index]][date], date])
            if not trades_log is None:
                trades_log.loc[len(trades_log)] = [date, prices.columns[index], "buy",
                                                   prices[prices.columns[index]][date], None]
        if a == 2:
            for i in range(len(self.portfolio)):
                if self.portfolio[i][0] == index:
                    x = prices.index.get_loc(date)
                    price_now = prices[self.portfolio[i][1]].iloc[x]
                    if price_now > self.portfolio[i][2]:
                        self.hits += 1
                    elif price_now < self.portfolio[i][2]:
                        self.miss += 1
                    if not trades_log is None:
                        trades_log.loc[len(trades_log)] = [date, self.portfolio[i][1], "sell", self.portfolio[i][2],price_now]
                    del self.portfolio[i]
                    break

    #Calculate next state given current state and update portfolio
    def _next_state(self,previous_state,next_state):
        holds = previous_state[-self.STOCK_COUNT:]
        index = int(self.action_index / 3)
        a = self.action_index % 3
        dupicate_stock = ""
        if a == 0:
            holds[index] = 1.0
            if self.verbose_logging:
                error=True
                duplicate=False
                for p in self.portfolio:
                    if p[0]==index:
                        if not error:
                            duplicate = True
                            dupicate_stock=p[1]
                            break
                        else:
                            error=False
                if error:
                    print "Error Buy existent: Time {} Index {}".format(self._time, index)
                if duplicate:
                    print "Duplicate: Time {} stock {}".format(self._time, dupicate_stock)

        elif a == 1:
            if self.verbose_logging:
                error = True
                for p in self.portfolio:
                    if p[0] == index:
                        error = False
                if error:
                    print "Error Hold with none: Time {} Index {}".format(self._time, index)
        elif a == 2:
            holds[index] = 0.0
            if self.verbose_logging:
                error = False
                for p in self.portfolio:
                    if p[0] == index:
                            error = True
                if error:
                    print "Error portofolio did not errase sell: Time {} Index {}".format(self._time, index)

        return np.append(next_state, holds)

    #Caluclate reward of date
    def _calculate_reward(self,date,prices):
        reward = 0
        if len(self.portfolio) == 0:
            return 0
        i = prices.index.get_loc(date)
        for p in self.portfolio:
                price_now = prices[prices.columns[p[0]]].iloc[i]
                price_next = prices[prices.columns[p[0]]].iloc[i + 1]
                reward += price_next / price_now - 1
        return reward/len(self.portfolio)

    #Get Return of spy in date
    def _get_benchmark_return(self,prices,date):
        i = prices.index.get_loc(date)
        price_now = prices["SPY"].iloc[i]
        price_next = prices["SPY"].iloc[i + 1]
        return  price_next / price_now - 1

    #Check if an action is valid
    def _check_valid_action(self,state,action_index):
        holds=state[-self.STOCK_COUNT:]
        index = int(action_index / 3)
        a = action_index % 3
        if a==0 and len(self.portfolio)>=self.MAX_PORTFOLIO_SIZE:
            return False
        if holds[index] == 0 and a > 0:
            return False
        if holds[index] == 1 and a == 0:
            return False
        return  True

    #Get a random but VALID action
    def _get_random_valid_action(self,state):

        # If portfolio full, just look for sell or hold of those stocks
        if len(self.portfolio)>=self.MAX_PORTFOLIO_SIZE:
            random_stock = random.choice(self.portfolio)
            random_action= random.choice([1,2])
            self.action_index= random_stock[0]*3+random_action

            return self.action_index

        while True:
            self.action_index = random.randrange(self.ACTIONS_COUNT)
            if self._check_valid_action(state, self.action_index):
                break

        return  self.action_index




    #Choose next action
    def _choose_next_action(self,state):
        new_action = np.zeros([self.ACTIONS_COUNT])
        self.action_index=0
        if random.random() <= self._probability_of_random_action:
            # choose an action randomly
                self.action_index = self._get_random_valid_action(state)

        else:
            # choose an action given our last state
            readout_t = self._session.run(self._output_layer, feed_dict={self._input_layer: [state]})[0]
            ordered_index= np.argsort(readout_t)
            for i in range(len(ordered_index)):
                self.action_index=ordered_index[len(ordered_index)-i-1]
                if self._check_valid_action(state, self.action_index):
                    break

        new_action[self.action_index] = 1

        return new_action

    #Tran model with n_iterations
    def _train(self,n_iterations,train_states,prices, fixed_length=False,save=True):
        self.EXPLORE_STEPS=int(n_iterations*0.9*(self.LENGTH_TRADE_MIN+self.LENGTH_TRADE_MAX)/2)
        self._time = 0
        train_data = pd.DataFrame(columns=["portafolio_return","spy_return","stocks", "hits","miss","total_loss","probability_of_random","time"])

        for trade_run in range(1,n_iterations+1):
            current_states = np.append(train_states[0][1:],np.zeros(self.STOCK_COUNT))
            self.portfolio = []
            portfolio_return=1
            spy_return = 1
            total_loss=0
            self.hits=0
            self.miss = 0
            if fixed_length:
                lenght_trade = len(train_states)-2
                start = 0
            else:
                lenght_trade=random.randint(self.LENGTH_TRADE_MIN,self.LENGTH_TRADE_MAX)
                start=random.randint(0, len(train_states)-lenght_trade-2)
            previous_states=[]
            actions=[]
            agents_expected_rewards=[]
            for i in range(start,start+lenght_trade+1):
                date= train_states[i][0]
                previous_state = current_states

                action = self._choose_next_action(previous_state)
                self._update_portfolio(prices,date)
                current_state = self._next_state(previous_state, train_states[i + 1][1:])
                reward = self._calculate_reward(date,prices)

                spy_return*=(1+self._get_benchmark_return(prices,date))
                portfolio_return *= (1+reward)


                # this gives us the agents expected reward for each action we might
                agents_reward_for_action = self._session.run(self._output_layer, feed_dict={self._input_layer: [current_state]})[0]
                agents_expected_reward = reward + self.FUTURE_REWARD_DISCOUNT * np.max(agents_reward_for_action)

                total_loss += (agents_expected_reward-np.max(agents_reward_for_action))*(agents_expected_reward-np.max(agents_reward_for_action))

                # store that these actions in these states lead to this reward
                previous_states.append(previous_state)
                actions.append(action)
                agents_expected_rewards.append(agents_expected_reward)

                # gradually reduce the probability of a random actionself.
                if self._probability_of_random_action > self.FINAL_RANDOM_ACTION_PROB:
                    self._probability_of_random_action -= \
                        (self.INITIAL_RANDOM_ACTION_PROB - self.FINAL_RANDOM_ACTION_PROB) / self.EXPLORE_STEPS


                self._time += 1

            self._session.run(self._train_operation, feed_dict={
                self._input_layer: previous_states,
                self._action: actions,
                self._target: agents_expected_rewards})


            train_data.loc[trade_run]=np.array([portfolio_return-1,spy_return-1,len(self.portfolio),self.hits,self.miss,total_loss,self._probability_of_random_action,datetime.datetime.now()])


            if self.verbose_logging:
                print "Trade: {} Time: {} Return: {} Stocks: {} Hits: {} Miss: {} Loss: {} Prob:{} Date:{}".format(
                    trade_run,
                    self._time,
                    portfolio_return - spy_return,
                    len(self.portfolio),
                    self.hits,
                    self.miss,
                    total_loss,
                    self._probability_of_random_action,
                    datetime.datetime.now())
        if save:
            self._saver.save(self._session, self._checkpoint_path + '/network', global_step=n_iterations)
            train_data.to_csv("results/train_results{}.csv".format(self._results_name), sep=';', decimal=",")


    #Perform a trade with trained model
    def _trade(self,test_states, prices, random_trader=False):
        self._time = 0
        test_data = pd.DataFrame(columns=["portafolio_return", "spy_return", "stocks", "hits", "miss", "total_loss"])
        trades_log = pd.DataFrame(columns=["date", "symbol", "action", "price_init", "price_end"])
        current_states = np.append(test_states[0][1:], np.zeros(self.STOCK_COUNT))
        self.portfolio = []
        portfolio_return = 1
        spy_return = 1
        total_loss = 0
        self.hits = 0
        self.miss = 0
        if random_trader:
            self._probability_of_random_action = 1
        else:
            self._probability_of_random_action=0
        for i in range(1,len(test_states)-1):
            date = test_states[i][0]
            previous_state = current_states
            action = self._choose_next_action(previous_state)
            self._update_portfolio(prices, date,trades_log)
            current_state = self._next_state(previous_state, test_states[i + 1][1:])
            reward = self._calculate_reward(date, prices)

            spy_return *= (1 + self._get_benchmark_return(prices, date))
            portfolio_return *= (1 + reward)


            # this gives us the agents expected reward for each action we might
            agents_reward_for_action = \
            self._session.run(self._output_layer, feed_dict={self._input_layer: [current_state]})[0]
            agents_expected_reward = reward + self.FUTURE_REWARD_DISCOUNT * np.max(agents_reward_for_action)

            total_loss += (agents_expected_reward - np.max(agents_reward_for_action)) * (
            agents_expected_reward - np.max(agents_reward_for_action))

            self._time += 1


            if self.verbose_logging:
                print "Time: {} Return over SPY: {} Stocks: {} Hits: {} Miss: {} Loss: {}".format(
                    self._time,
                    portfolio_return - spy_return,
                    len(self.portfolio),
                    self.hits,
                    self.miss,
                    total_loss)
            if not random_trader:
                test_data.loc[self._time] = np.array(
                [portfolio_return - 1, spy_return - 1, len(self.portfolio), self.hits, self.miss, total_loss])

        if not random_trader:
            test_data.to_csv("results/test_results{}.csv".format(self._results_name), sep=';',decimal=",")
            trades_log.to_csv("results/trades_log{}.csv".format(self._results_name), sep=';', decimal=",")
        return (portfolio_return-1), (spy_return-1)

    #Save model
    def save(self,model_name):
        self._saver.save(self._session, self._checkpoint_path + '/'+model_name)


# Gets Saved Test Data
def getTestData():
    print "Loading Test Data"
    test_states= pd.read_pickle("data/test.pkl")
    print "Loading Prices"
    prices = pd.read_pickle("data/prices.pkl")
    return  test_states,prices

# Gets Saved Data Train + Test
def getAllData():
    print "Loading Train Data!"
    train_states = pd.read_pickle("data/train.pkl")
    test_states, prices = getTestData()
    nfeatures=len(train_states["State"][0])
    nstocks=len(prices.columns)
    print "Loading Done! {} features and {} stocks".format(nfeatures,nstocks)
    return  train_states,test_states,prices,nfeatures,nstocks


if __name__ == '__main__':

    #Choose mode
    mode=2

    print "Starting {}".format(datetime.datetime.now())
    if mode==1: #Perform trade from last saved model
        test_states, prices = getTestData()
        nfeatures = len(test_states["State"][0])
        nstocks = len(prices.columns)
        trader = DeepQTrader(nstocks * 3, nfeatures - 1 + nstocks, nstocks,1,0.1, verbose_logging=True, restore=True)
        print "Testing {}".format(datetime.datetime.now())
        trader._trade(test_states["State"].values, prices)
    elif mode==2: #Perform a simple train and trade session from stored data
        n_trades = 400
        n_layers = 1
        learning_rate = 0.001
        n_days=6
        window_size=200
        prices = pd.read_pickle("data/prices.pkl")
        train_states, test_states = preprocess.CalculateStates(prices, n_days, ['adjsma{}'.format(window_size), 'bb{}'.format(window_size)], '2014-11-22', False)
        test_states=test_states.head(88)
        nfeatures = len(test_states["State"][0])
        nstocks = len(prices.columns)

        trader = DeepQTrader(nstocks*3,nfeatures-1+nstocks,nstocks,n_layers,learning_rate,verbose_logging=True,restore=False,results_name="_ntrades{}".format(n_trades))
        print "Training {}".format(datetime.datetime.now())
        trader._train(n_trades,train_states["State"].values,prices)
        print "Testing {}".format(datetime.datetime.now())
        trader._trade(test_states["State"].values,prices)

    elif mode == 3: #Perform iteration to find the best candidates in first iteration
        for i in range(1,8):
            prices = pd.read_pickle("data/prices.pkl")
            train_states, test_states=preprocess.CalculateStates(prices, i, ['adjsma20', 'bb20'], '2016-01-01',False)
            nfeatures = len(test_states["State"][0])
            nstocks = len(prices.columns)
            cum_trades = 0
            for n_trades in [200, 300,400,500,600,700]:
                cum_trades += n_trades
                trader = DeepQTrader(nstocks * 3, nfeatures - 1 + nstocks, nstocks,2,0.1, verbose_logging=True, restore=False,
                                     results_name="2hidden_ndays{}_ntrades{}".format(i,cum_trades))
                print "Training {}".format(datetime.datetime.now())
                trader._train(cum_trades, train_states["State"].values, prices)
                print "Testing {}".format(datetime.datetime.now())
                trader._trade(test_states["State"].values, prices)

    elif mode == 4: # Perform second iteration of best performers in first iteration
        for c in [[5, 300, 1],[6, 400, 1],[1, 500, 2],[4, 700, 2]]:
            for i in [10,20,50,200]:
                adj = 'adjsma{}'.format(i)
                bb = 'bb{}'.format(i)
                prices = pd.read_pickle("data/prices.pkl")
                train_states, test_states = preprocess.CalculateStates(prices, c[0], [adj, bb], '2016-01-01', False)
                nfeatures = len(test_states["State"][0])
                nstocks = len(prices.columns)

                for lr in [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]:
                    trader = DeepQTrader(nstocks * 3, nfeatures - 1 + nstocks, nstocks, c[2], lr, verbose_logging=True,
                                         restore=False,
                                         results_name="{}hidden_ndays{}_ntrades{}_lr{}_w{}".format(c[2], c[0], c[1], lr, i))
                    print "Training {}".format(datetime.datetime.now())
                    trader._train(c[1], train_states["State"].values, prices)
                    print "Testing {}".format(datetime.datetime.now())
                    trader._trade(test_states["State"].values, prices)

    elif mode == 5:  # Perform Roll Forward Cross Validation
        i = 1
        for c in [[6, 600, 1, 20, 0.01], [6, 600, 1, 200, 0.001], [4, 700, 2, 20, 0.0001], [4, 700, 2, 200, 0.000001]]:
            k_folds_results = pd.DataFrame(
                columns=["days", "trades", "layers", "window", "learning_rate", "return", "spy"])
            adj = 'adjsma{}'.format(c[3])
            bb = 'bb{}'.format(c[3])
            prices = pd.read_pickle("data/prices.pkl")
            train_states, test_states = preprocess.CalculateStates(prices, c[0], [adj, bb], '2200-01-21', False)
            nfeatures = len(train_states["State"][0])
            nstocks = len(prices.columns)
            k_folds = 18
            folds_lenght = int(len(train_states) / k_folds)
            restore = False
            states_values = train_states["State"].values
            ntrains = int(c[1] / k_folds)
            trader = DeepQTrader(nstocks * 3, nfeatures - 1 + nstocks, nstocks, c[2], c[4], verbose_logging=True,
                                 restore=restore, results_name="final{}".format(i))
            for k in range(k_folds - 1):
                k_train_states = states_values[k * folds_lenght:(k + 1) * folds_lenght - 1]
                k_test_states = states_values[(k + 1) * folds_lenght:(k + 2) * folds_lenght - 1]
                print "Training model {} k {} {}".format(i,k,datetime.datetime.now())
                trader._train(ntrains, k_train_states, prices, fixed_length=True, save=False)
                print "Testing model {} k {} {}".format(i,k,datetime.datetime.now())
                trade_return, spy = trader._trade(k_test_states, prices)
                k_folds_results.loc[len(k_folds_results)] = [c[0], c[1], c[2], c[3], c[4], trade_return, spy]
            k_folds_results.to_csv("results/k_fold_{}.csv".format(i), sep=';', decimal=",")
            trader.save("model{}".format(i))
            i += 1

    elif mode == 6: #Perform Random Trader
        random_trader_results = pd.DataFrame(
            columns=[ "return", "spy"])
        prices = pd.read_pickle("data/prices.pkl")
        train_states, test_states = preprocess.CalculateStates(prices, 1, ['adjsma20', 'bb20'], '2200-01-21', False)
        nfeatures = len(train_states["State"][0])
        nstocks = len(prices.columns)
        k_folds = 18
        folds_lenght = int(len(train_states) / k_folds)
        states_values = train_states["State"].values
        trader = DeepQTrader(nstocks * 3, nfeatures - 1 + nstocks, nstocks, 1,0.1, verbose_logging=False, restore=False)
        result=0
        for k in range(k_folds - 1):
            k_test_states = states_values[(k + 1) * folds_lenght:(k + 2) * folds_lenght - 1]
            result,spy=trader._trade(k_test_states, prices,random_trader=True)
            print("Random Trader Avarage Return in fold {} was {} and SPY was {}".format(k,result,spy))
            print("from {} to {}".format(states_values[(k + 1) * folds_lenght][0],states_values[(k + 2) * folds_lenght - 1][0]))
            random_trader_results.loc[len(random_trader_results)] = [result, spy]
        #random_trader_results.to_csv("results/random_trader.csv", sep=';', decimal=",")




    print "Done {}".format(datetime.datetime.now())