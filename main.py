import pandas as pd
from scipy.stats import pearsonr, chi2_contingency
from xgboost import XGBRegressor
from utils import *


data_path = './data/'
max_deviation = 10
plot_path = './plots/'

if __name__ == '__main__':
	
    x_train = pd.read_csv(data_path+'xtrain.csv')
    x_test = pd.read_csv(data_path+'xtest.csv')
    y_train = pd.read_csv(data_path+'ytrain.csv')
    y_test = pd.read_csv(data_path+'ytest.csv')

    test = pd.Series(y_test.iloc[:,0])

    pred1 = neural_network(x_train, y_train, x_test, y_test)
    plt.scatter(y_test, pred1)
    plt.title('Neural Network Predictions')
    p1, p2 = [300, 1000], [300, 1000]
    plt.plot(p1, p2, color ='red') 
    plt.savefig(plot_path+'NN_predictions.png')
    plt.show()

    # calculate Pearson's correlation
    corr, _ = pearsonr(test, pred1)
    print('Pearsons correlation NEURAL NETWORK: %.3f' % corr)
    rmse= np.sqrt(np.square(test-pred1))
    print('Avg RMSE NEURAL NETWORK:', np.average(rmse))
    diff= np.abs(test-pred1)
    res = diff<max_deviation
    acc = np.count_nonzero(res)/len(res)
    print('Accuracy NEURAL NETWORK: %.3f' % acc)


    # fit model no training data
    model2 = XGBRegressor()
    model2.fit(x_train, y_train)
    yhat = model2.predict(x_test)

    plt.scatter(y_test, yhat)
    plt.title('XGBOOST Predictions')
    p1, p2 = [300, 1000], [300, 1000]
    plt.plot(p1, p2, color ='red') 
    plt.savefig(plot_path+'XGBOOST_predictions.png')
    plt.show()


    # calculate Pearson's correlation
    corr, _ = pearsonr(test, yhat)
    print('Pearsons correlation XGBOOST: %.3f' % corr)
    rmse= np.sqrt(np.square(test-yhat))
    print('Avg RMSE XGBOOST:', np.average(rmse))
    diff= np.abs(test-yhat)
    res = diff<max_deviation
    acc = np.count_nonzero(res)/len(res)
    print('Accuracy XGBOOST: %.3f' % acc)
