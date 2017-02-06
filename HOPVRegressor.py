from paramRegressor import *


def TwoSig(x):
    return float("{0:.2f}".format(x))

def ThreeSig(x):
    return float("{0:.3f}".format(x))

def saveNewResults(yType,funcType, fpType, regType, best_metric,best_grid, best_estimator,best_predict,cum_score):
	dictionary = {}
	dictionary['r2']= ThreeSig(best_metric[0])
	dictionary['r']= best_metric[1]
	dictionary['MSE']= best_metric[2]
	dictionary['RMSE']= best_metric[3]
	dictionary['MAE']= best_metric[4]
	dictionary['MAPE']= best_metric[5]
	dictionary['cum_score'] = cum_score
	dictionary['best_grid'] = best_grid
	dictionary['best_estimator'] = best_estimator
	dictionary['predicted'] = best_predict

	fileName = yType+"_"+fpType+"_"+regType+"_"+funcType+"_"+'supervised'+str(cum_score)
	saveData(dictionary,fileName,"regMetrics")
	print best_metric[0],best_estimator
	print "result saved in ",fileName


def modRegressor(X_full,y_full,estimator,bestParam):

    estimator.set_params(**bestParam)
    y_predict = []
    y_true = y_full
    loo = cross_validation.LeaveOneOut(len(X_full))
    count = 0
    for train_index, test_index in loo:
        count += 1
        X_train, X_test = X_full[train_index], X_full[test_index]
        y_train, y_test = y_full[train_index], y_full[test_index]
        predicted = estimator.fit(X_train, y_train).predict(X_test)
        print "[",count,"] estimator running ....."
        y_predict += [predicted[0]]

    score = r2_score(y_true,y_predict)
    R = r_score(y_true,y_predict)
    MSE = mean_squared_error(y_true, y_predict)
    RMSE = root_mean_squared_error(y_true,y_predict)
    MAE = mean_absolute_error(y_true, y_predict)
    MAPE = mean_absolute_percentage_error(y_true, y_predict)

    best_grid = bestParam
    best_estimator = estimator
    best_metric = [score, R, MSE, RMSE, MAE,MAPE]
    best_predict = y_predict

    return best_estimator,best_grid,best_metric,best_predict


def runCombinedFPSupervisedGrid(yType,funcType,fpType="AtomTopo",regType="randomForest"):

    regType = 'randomForest'
    print fpType
    for idx in range(24):
        print
        print 'Step',idx+1,'of the supervised feature selection'
        print '_________________________________________'
        print
        if idx ==0:
            X_full = loadData("X_full_selected"+fpType,"FP")
        else:
            X_full = loadData("X_full_"+fpType+str(idx),"supervised")
        y_true = array(loadData(yType+'_'+funcType,"FP"))
        bestParam = loadData('selectedAtomTopo_randomForest_B3LYP_supervised'+str(idx),'supervised')['best_grid']

        estimator = getEstimator(regType)

        best_estimator,best_grid,best_metric,best_predict = modRegressor(X_full,y_true,estimator,bestParam)

        if idx<13:
            featureReduce = 200
        else:
            featureReduce = 20

        X_full_reduced = supervisedTransform(X_full, best_estimator,featureReduce)
        print len(X_full_reduced),len(X_full_reduced[0])
        saveData(X_full_reduced,'X_full_'+fpType+str(idx+1),'supervised')
        saveNewResults(yType,funcType, fpType, regType, best_metric,best_grid, best_estimator,best_predict,idx)

    return

def supervisedTransform(X_train,estimator,featureReduce):
    features = estimator.feature_importances_
    std = np.std([tree.feature_importances_ for tree in estimator.estimators_],
                 axis=0)
    #This line sorts the features by importance but does not preserve the order
    indices = np.argsort(features)[::-1]
    reducedIndices = indices[:len(indices)-featureReduce]
    X_train_reduced = []
    for row in X_train:
        row_reduced = [row[i] for i in reducedIndices]
        X_train_reduced += [row_reduced]
    print 'reducing features'
    saveData(features,'features','newFP')
    saveData(std,'std','newFP')

    return array(X_train_reduced)

def runModifiedSupervisedGrid(yType,funcType,fpType="AtomPair",regType="randomForest"):

    regType = 'randomForest'
    print fpType
    for idx in range(21):
        print
        print 'Step',idx+1,'of the supervised feature selection'
        print '_________________________________________'
        print
        # if idx ==0:
        #     X_full = loadData("X_full_"+fpType,"FP")
        # else:
        X_full = loadData("X_full_"+fpType+str(idx),"newFP")
        y_true = array(loadData(yType+'_'+funcType,"FP"))
        bestParam = loadData('selectedAtomTopo_randomForest_B3LYP_supervised'+str(idx),'supervised')['best_grid']

        estimator = getEstimator(regType)

        best_estimator,best_grid,best_metric,best_predict = modRegressor(X_full,y_true,estimator,bestParam)

        if idx<9:
            featureReduce = 200
        else:
            featureReduce = 20

        X_full_reduced = supervisedTransform(X_full, best_estimator,featureReduce)
        print len(X_full_reduced),len(X_full_reduced[0])
        saveData(X_full_reduced,'X_full_'+fpType+str(idx+1),'newFP')
        saveNewResults(yType,funcType, fpType, regType, best_metric,best_grid, best_estimator,best_predict,idx)

    return

    return best_estimator,best_grid,best_metric,best_predict

if __name__ == "__main__":
    runModifiedSupervisedGrid('HOMO','B3LYP')
