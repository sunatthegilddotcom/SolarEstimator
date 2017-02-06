from sys import argv
import pickle,warnings,scipy
from scipy.stats import randint as sp_randint
from paulRegressor import *
import numpy as np
from numpy import array,reshape
from random import shuffle

warnings.filterwarnings("ignore")
NEG = -100.34430435082e+35
cum_scores = [0.6,0.7,0.75,0.8,0.85,0.9,0.95,0.99]
# allParamsGrid = {"randomForest":{'min_samples_split':[5, 10, 15, 20], 'n_estimators':[100, 200], 'max_features':[0.25, 0.33], 'max_depth':[10 ,25], 'n_jobs':[-1]},
# 	"extraTrees":{'n_estimators':[3,5,10],'n_jobs':[-1]},
# 	"rbfSVM":{'gamma': [1e-3, 1e-4],
# 	                     'C': [1, 10, 100, 1000]},
# 	"linearSVM":{'C': [1, 10, 100, 1000]},
# 	"ridge":{'alpha':[0.001],'fit_intercept':[True,False],'normalize':[True],'solver':['svd'],'tol':[0.0001]},
# 	"dummy":{'strategy':['mean','median']},
# 	"LinearRegression":{},"NNGarrotteRegression":{},"KernelRegression":{},"AdaBoost":{},"Bagging":{},
# 	'SGDRegression': {'penalty':['l1','l2','elasticnet',None],'l1_ratio':[0.01,0.10,0.20,0.80]},
# 	"KNeighborsRegression":{'n_neighbors':[2,5,10],'weights':['uniform','distance'],'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']},
# 	"MultiLasso":{'alpha':[0.01,0.1,1.0,10.0]},
# 	"lasso":{'alpha':[0.01,0.1,1.0,10.0]},
# 	"DecisionTree":{'max_depth':[5,8,11]},
# 	"MultiElasticNet":{'alpha':[0.5,1,2],'l1_ratio':[0,0.5,1.0],'normalize':[True,False],'warm_start':[True,False]}}#'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}}

reducedGrid = {"randomForest":{'min_samples_split':[5], 'n_estimators':[100], 'max_features':[0.33],
                               'max_depth':[25], 'n_jobs':[-1]}}

paramsGrid = {"randomForest":{'min_samples_split':[5, 10, 15, 20], 'n_estimators':[100, 200], 'max_features':[0.25, 0.33], 'max_depth':[10 ,25], 'n_jobs':[-1]},
	"extraTrees":{'n_estimators':[2,3],'n_jobs':[-1]},
	"rbfSVM":{'gamma': [1e-3, 1e-4],
	                     'C': [1, 10, 100, 1000]},
	"linearSVM":{'C': [1, 10, 100, 1000]},
	"ridge":{'alpha':[0.001],'fit_intercept':[True,False],'normalize':[True],'solver':['svd'],'tol':[0.0001]},
	"dummy":{'strategy':['mean','median']},
	"LinearRegression":{'fit_intercept':[True,False],'normalize':[True,False]},	"NNGarrotteRegression":{},"KernelRegression":{},
	"KernelRidge":{'alpha':[1,10]},
	"AdaBoost":{},"Bagging":{},
	'SGDRegression': {'penalty':['l1','l2','elasticnet',None],'l1_ratio':[0.01,0.10,0.20,0.80]},
	"KNeighborsRegression":{'n_neighbors':[2,5,10],'weights':['uniform','distance'],'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']},
	"MultiLasso":{'alpha':[0.01,0.1,1.0,10.0]},
	"lasso":{'alpha':[0.01,0.1,1.0,10.0]},
	"DecisionTree":{'max_depth':[5,8,11]},
	"MultiElasticNet":{'alpha':[0.5,1,2],'l1_ratio':[0,0.5,1.0],'normalize':[True,False],'warm_start':[True,False]}}


def mean_absolute_percentage_error(y_true, y_pred):
	'''
	scikit(sklearn) does not have support for mean absolute percentage error MAPE.
	This is because the denominator can theoretically be 0 and so the value would be undefined.
	So this is our implementation
	'''
	y_true = check_arrays(y_true)
	y_pred = check_arrays(y_pred)

	return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def root_mean_squared_error(y_true,y_pred):
	return mean_squared_error(y_true, y_pred)**0.5

def r_score(y_true,y_pred):
	R2 = r2_score(y_true,y_pred)
	if R2 >0:
		return R2**0.5
	else:
		return None

def firstIndexAbove(alist,x):
    return [ n for n,i in enumerate(alist) if float(i)>x ][0]

def TwoSig(x):
    return float("{0:.2f}".format(x))

def ThreeSig(x):
    return float("{0:.3f}".format(x))

def findNumComponents(X_train_std):
    cov_mat = np.cov(X_train_std.T)
    eigen_vals,eigen_vecs = np.linalg.eig(cov_mat)
    tot = sum(eigen_vals)
    num = len(eigen_vals)+1
    var_exp = [(i/tot) for i in sorted(eigen_vals,reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    List = []

    for score in cum_scores:
        List +=[firstIndexAbove(cum_var_exp,score)]
    return List


def pca_transform(X_train):
	sc = StandardScaler()
	X_list = []
	X_train_std = sc.fit_transform(X_train)
	components = findNumComponents(X_train_std)
	for component in components:
		pca = PCA(n_components=component)
		X_train_pca = pca.fit_transform(X_train_std)
		X_list +=[np.array(X_train_pca)]

	return X_list

def rca_transform(X_train):
    sc = StandardScaler()
    X_list = []
    X_train_std = sc.fit_transform(X_train)
    components = findNumComponents(X_train_std)
    for component in components:
        rca = RCA(n_components=component)
        X_train_rca = rca.fit_transform(X_train_std)

        X_list +=[np.array(X_train_rca)]

    return X_list

def sparse_pca_transform(X_train):
	sc = StandardScaler()
	X_list = []
	X_train_std = sc.fit_transform(X_train)
	components = findNumComponents(X_train_std)
	for component in components:
		print "for loop"
		pca = MiniBatchSparsePCA(n_components=component,n_jobs=-1,batch_size=10)#SparsePCA(n_components=component)
		X_train_pca = pca.fit_transform(X_train_std)
		X_list +=[np.array(X_train_pca)]

	return X_list

def truncatedSVD_transform(X_train):
    sc = StandardScaler()
    X_list = []
    X_train_std = sc.fit_transform(X_train)
    components = findNumComponents(X_train_std)
    for component in components:
        tsvd = TruncatedSVD(n_components=component)
        X_train_tsvd = tsvd.fit_transform(X_train_std)

        X_list +=[np.array(X_train_tsvd)]

    return X_list

def str2bool(v):
	'''
	This converts a Boolean response from string to boolean format.
	For e.g. when we are taking in standard input for a value, it would be a string.
	However, because we would like to use it as a boolean variable, we use this function.
	'''
	return v.lower() in ("yes", "true", "t", "1")

def sqr(x):
	'''
	It just gives square. A function to make our life easy
	'''
	return pow(x,2)

def mse(x,y):
	'''
	Personal implementation of mean squared error
	Not using it currently
	'''
	MSE = 0
	for i in range(len(x)):
		MSE += sqr(x[i]-y[i])
	return MSE

def savePkl(obj, name,path="FPkls"):
    '''
    This saves a object into a pickle file. In our case, it is generally a DICTIONARY object.
    '''

    with open(path+"/"+name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def saveData(obj, name, path="FP"):
	savePkl(obj,name,path)


def loadData(name,path="FP"):
	'''
	This loads a pickle file and returns the content which is a DICTIONARY object in our case.
	'''
	if ".pkl" in name:
		name = name.split(".pkl")[0]
	if "/" in name:
		name = name.split("/",1)[1]
	with open(path+"/"+name + '.pkl', 'rb') as f:
		return pickle.load(f)

def load(name,path="FP"):
	'''
	This loads a pickle file and returns the content which is a DICTIONARY object in our case.
	'''
	if ".pkl" in name:
		name = name.split(".pkl")[0]
	if "/" in name:
		name = name.split("/",1)[1]
	with open(path+"/"+name + '.pkl', 'rb') as f:
		return pickle.load(f)

def getValuesSimple(prop,estimator):
	X_full = load("X_full_"+prop)
	y_full = load("y_full_"+prop)
	loo = cross_validation.LeaveOneOut(len(X_full))
	y_predict = []
	y_true = y_full

	print "Loaded"
	name = "y_"+argv[2]+"_"+prop+"_predict"
	f = open(name+".txt","w")
	for train_index, test_index in loo:
	 	# print("TRAIN:", train_index, "TEST:", test_index)
	 	X_train, X_test = X_full[train_index], X_full[test_index]
	 	y_train, y_test = y_full[train_index], y_full[test_index]
		predicted = estimator.fit(X_train, y_train).predict(X_test)
	 	y_predict += [predicted]
		f.write(str(predicted)+"\n")
	 	print("One row done")
	f.close()

	MSE = mean_squared_error(y_true, y_predict)
	R2 = r2_score(y_true, y_predict)
	MAE = mean_absolute_error(y_true,y_predict)
	MAPE = mean_absolute_percentage_error(y_true,y_predict)
	print(MSE)
	print(R2)
	print(MAE)
	print(MAPE)
	# print (myMSE)
	savePkl(y_predict,name)
	f= open(name+".dat","w")
	f.write(str(MSE)+"/n")
	f.write(str(R2)+"/n")
	f.write(str(MAE)+"/n")
	f.write(str(MAPE)+"/n")
	f.close()

def getValues(fingerprintType,learner):
	prop = fingerprintType
	estimator = getEstimator(learner)
	X_full = load("X_full_"+prop,"FP")
	y_full = load("y_full","FP")

	#X_full = X_full.reshape(-1,1)
	y_predictArr = []
	y_predict = []
	loo = cross_validation.LeaveOneOut(len(X_full))
	for train_index, test_index in loo:
	 	X_train, X_test = X_full[train_index], X_full[test_index]
	 	y_train, y_test = y_full[train_index], y_full[test_index]
		predicted = estimator.fit(X_train, y_train).predict(X_test)
	 	y_predict += [predicted[0]]

	y_true = y_full
	MSE = mean_squared_error(y_true, y_predict)
	R2 = r2_score(y_true, y_predict)
	MAE = mean_absolute_error(y_true,y_predict)
	MAPE = mean_absolute_percentage_error(y_true,y_predict)
	regressionCoeff = [MSE, MAE, MAPE,R2]

	name = "y_"+argv[2]+"_"+prop+"_predict"
	# savePkl(y_predict,name+"_values","regValues")
	# savePkl(regressionCoeff,name+"_coeff","regCoeff")

	print prop,estimator,"complete"

def sortShuffleSplit(data, split_percentage,ifShuffle):

	train,validate = [],[]
	trainX,validateX = [],[]
	trainY,validateY = [],[]

	jump = int(1/split_percentage)
	data = sorted(data, key=lambda row: row[1])

	for i in range(len(data)):

		if i%jump ==0:
			validate += [data[i]]
		else:
			train += [data[i]]

	if ifShuffle:
		print "shuffling"
		shuffle(train)
		shuffle(validate)
		shuffle(train)
		shuffle(validate)
	else:
		print "not shuffling"

	for i in range(len(train)):
		trainX += [train[i][0]]
		trainY += [train[i][1]]

	for i in range(len(validate)):
		validateX += [validate[i][0]]
		validateY += [validate[i][1]]

	return trainX,validateX,trainY,validateY

def gridTrainData(gridType,estimator,parameters,trainX,trainY,n_jobs=-1):
	loo = LeaveOneOut(len(trainX))
	if gridType == "grid":
		reg = GridSearchCV(estimator, parameters, n_jobs = n_jobs, cv=loo)
	else:
		reg = RandomizedSearchCV(estimator, parameters, cv=loo)
	print reg

	reg.fit(trainX,trainY)
	reg_train = reg.best_estimator_
	reg_train.fit(trainX,trainY)
	return reg_train

def gridSearchRegressor(fingerprintType,regType,gridType,parameters,split_percentage=0.2,n_jobs=-1,ifShuffle=True,degree=3):
	prop = fingerprintType
	print regType
	estimator = getEstimator(regType,degree)
	X_full = load("X_full_"+prop,"FP")
	y_full = load("y_full","FP")
	y_predict = []

	data = zip(X_full,y_full)

	trainX, validateX,trainY,validateY = sortShuffleSplit(data,split_percentage,ifShuffle)

	reg_train = gridTrainData(gridType, estimator,parameters,trainX,trainY	,n_jobs)
	y_predict = reg_train.predict(validateX)
	y_true = validateY

	MSE = mean_squared_error(y_true, y_predict)
	R2 = r2_score(y_true, y_predict)
	MAE = mean_absolute_error(y_true, y_predict)
	MAPE = mean_absolute_percentage_error(y_true, y_predict)
	regressionCoeff = [MSE, MAE, MAPE,R2]

	print "MSE",MSE
	print "MAE",MAE
	print "MAPE",MAPE
	print "R2",R2
	print reg_train

def oldMain(args):
	fingerprintType = args[1]
	regType = args[2]
	n_jobs = int(args[3])
	gridType = "grid"
	if len(args)>4:
		split_percentage = float(args[4])
	else:
		split_percentage=0.2

	if len(args)>5:
		ifShuffle = str2bool(args[5])
	else:
		ifShuffle = True

	if len(args)>6:
		degree = int(args[6])
	else:
		degree = 3

	paramsRandom = {"randomForest":{"max_depth": [3, None],"max_features": sp_randint(1, 11), "min_samples_split": sp_randint(1, 11), "min_samples_leaf": sp_randint(1, 11),"bootstrap": [True, False]},
			"rbfSVM":{'C': scipy.stats.expon(scale=100), 'gamma': scipy.stats.expon(scale=.1)},
			"linearSVM":{'C': sp_randint(1, 1000)}}


	paramsGrid = {"randomForest":{'min_samples_split':[5, 10, 15, 20], 'n_estimators':[100, 200], 'max_features':[0.25, 0.33], 'max_depth':[10 ,25], 'random_state':[1983]},
	"extraTrees":{'n_estimators':[2]},
	"rbfSVM":{'gamma': [1e-3, 1e-4],
	                     'C': [1, 10, 100, 1000]},
	"linearSVM":{'C': [1, 10, 100, 1000]},
	"ridge":{'alpha':[0.001],'fit_intercept':[True,False],'normalize':[True],'solver':['svd'],'tol':[0.0001]},
	"dummy":{'strategy':['mean','median']},
	"LinearRegression":{},
	"NNGarrotteRegression":{},
	"KernelRegression":{},
	"AdaBoost":{},
	"Bagging":{},
	'SGDRegression': {'penalty':['l1','l2','elasticnet',None],'l1_ratio':[0.01,0.10,0.20,0.80]},
	"KNeighborsRegression":{'n_neighbors':[2,5,10],'weights':['uniform','distance'],'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']},
	"MultiLasso":{'alpha':[0.01,0.1,1.0,10.0]},
	"lasso":{'alpha':[0.01,0.1,1.0,10.0]},
	"DecisionTree":{'max_depth':[2,5,8]},
	"MultiElasticNet":{'alpha':[0.5,1,2],'l1_ratio':[0,0.5,1.0],'normalize':[True,False],'warm_start':[True,False]}}#'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}}


	if "random" in gridType:
		params = paramsRandom
	else:
		params = paramsGrid

	parameters = params[regType]
	gridSearchRegressor(fingerprintType,regType,gridType,parameters,split_percentage,n_jobs,ifShuffle,degree)

def leaveOneOut(X_full,y_full):
	y_predict = []
	loo = cross_validation.LeaveOneOut(len(X_full))
	for train_index, test_index in loo:
	 	# print("TRAIN:", train_index, "TEST:", test_index)
	 	X_train, X_test = X_full[train_index], X_full[test_index]
	 	y_train, y_test = y_full[train_index], y_full[test_index]
	 	y_predict += [estimator.fit(X_train, y_train).predict(X_test)]
	 	print("One row done")
	return y_predict

def getEstimator(regressor,degree=3):
	'''
	This calls regressor functions from paulRegressor based on
	Note: paulRegressor is a module which loads all the required regressor modules from various scikit or scikit-based modules
	'''

	if regressor=="lasso" or regressor=="Lasso":
		estimator = Lasso(alpha = 0.1)#RandomForestRegressor(random_state=0, n_estimators=100)\
	elif "MultiLasso" in regressor:
		estimator = MultiLasso()
	elif regressor=="ridge" or regressor=="Ridge":
		estimator = Ridge()#(alphas=[0.1, 1.0, 10.0])
	elif "SGDRegression" in regressor:
		estimator = SGDRegressor()
	elif "NNGarrotteRegression" in regressor:
		estimator = NNGarrotteRegressor()
	elif "KernelRegression" in regressor:
		estimator = KernelRegressor()
	elif regressor=="LinearRegression":
		estimator = LinearRegression()
	elif "NonLinearRegression" in regressor:
		estimator = NonLinearRegressor(degree)
	elif "KNeighborsRegression" in regressor:
		estimator = KNeighborsRegressor()
	elif "randomForest" in regressor or "RandomForest" in regressor:
		estimator = RandomForestRegressor()
	elif "extraTrees" in regressor or "ExtraTrees" in regressor:
		estimator = ExtraTreesRegressor()
	elif "rbfSVM" in regressor or "RBFSVM" in regressor:
		estimator = SVR(kernel="rbf")
	elif "linearSVM" in regressor or "LinearSVM" in regressor:
		estimator = SVR(kernel="linear")
	elif "polySVM" in regressor or "PolySVM" in regressor:
		estimator = polySVR()
	elif regressor=="ElasticNet":
		estimator = ElasticNet()
	elif "MultiElasticNet" in regressor:
		estimator = MultiElasticNet()
	elif "gradientBoost" in regressor or "GradientBoost" in regressor:
		estimator = gradientBoost()
	elif "AdaBoost" in regressor:
		estimator = AdaBoostRegressor()
	elif "Bagging" in regressor:
		estimator = BaggingRegressor()
	elif "DecisionTree" in regressor:
		estimator = DecisionTreeRegressor()
	elif "KernelRidge" in regressor:
		estimator = KernelRidge()
	elif "dummy" in regressor:
		estimator = DummyRegressor()

	return estimator

def saveResults(funcType, fpType, regType, best_metric,best_grid, best_estimator,best_predict,scores,standard=False):
	dictionary = {}
	dictionary['r2']= best_metric[0]
	dictionary['r']= best_metric[1]
	dictionary['MSE']= best_metric[2]
	dictionary['RMSE']= best_metric[3]
	dictionary['MAE']= best_metric[4]
	dictionary['MAPE']= best_metric[5]

	dictionary['best_grid'] = best_grid
	dictionary['best_estimator'] = best_estimator
	dictionary['predicted'] = best_predict
	if standard:
		fileName = fpType+"_"+regType+"_"+funcType+"_standard"
	else:
		fileName = fpType+"_"+regType+"_"+funcType
	savePkl(dictionary,fileName,"regMetrics")
	print best_metric[0],best_estimator
	print "scores",scores
	print "result saved in ",fileName

def GridSearchNoCV(funcType, fpType, regType,standard=False):
	X_full = loadData("X_full_"+fpType,"FP")
	if standard:
		X_full = StandardScaler().fit_transform(X_full)
		print 'standard'
	y_full = array(loadData("y_full_"+funcType,"FP"))
	scores = []
	estimator = getEstimator(regType)
	best_score = NEG
	best_grid = None
	best_predict = None
	best_estimator = None
	best_metric = [best_score,NEG,NEG,NEG]
	grid = paramsGrid[regType]

	for g in ParameterGrid(grid):
		estimator.set_params(**g)
		print estimator
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
		print score
		scores += [score]

		if score >= best_score:
			best_score = score
			R = r_score(y_true,y_predict)
			MSE = mean_squared_error(y_true, y_predict)
			RMSE = root_mean_squared_error(y_true,y_predict)
			MAE = mean_absolute_error(y_true, y_predict)
			MAPE = mean_absolute_percentage_error(y_true, y_predict)
			best_grid = g
			best_estimator = estimator
			best_metric = [score, R, MSE, RMSE, MAE,MAPE]
			best_predict = y_predict

	saveResults(funcType, fpType, regType, best_metric,best_grid, best_estimator,best_predict,scores,standard)

	return best_metric,best_grid, best_estimator,best_predict

def saveNewResults(funcType, fpType, regType, best_metric,best_grid, best_estimator,best_predict,cum_score,pca):
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
	if 'tsvd' in pca:
		fileName = fpType+"_"+regType+"_"+funcType+"_"+'tsvd'+str(cum_score)
	elif 'sparsepca' in pca:
		fileName = fpType+"_"+regType+"_"+funcType+"_"+'sparse_pca'+str(cum_score)
	elif 'rca' in pca:
		fileName = fpType+"_"+regType+"_"+funcType+"_"+'rca'+str(cum_score)
	elif 'supervised' in pca:
		fileName = fpType+"_"+regType+"_"+funcType+"_"+'supervised'+str(cum_score)
	else:
		fileName = fpType+"_"+regType+"_"+funcType+"_"+'pca'+str(cum_score)

	savePkl(dictionary,fileName,"regMetrics")
	print best_metric[0],best_estimator
	print "result saved in ",fileName

def GridSearchPCA(funcType, fpType, regType,pca='pca'):
	X_load = loadData("X_full_"+fpType,"FP")
	y_full = array(loadData("y_full_"+funcType,"FP"))

	#cum_scores = [0.9]#[0.3,0.4,0.5,0.6,0.7,0.8,0.9]

	estimator = getEstimator(regType)

	grid = paramsGrid[regType]

	if 'tsvd' in pca:
		print 'Transforming Truncated SVD.....'
		transformedXs = truncatedSVD_transform(X_load)
	elif 'sparsepca' in pca:
		print 'Transforming Sparse PCA.....'
		transformedXs = sparse_pca_transform(X_load)
	elif 'rca' in pca:
		print 'Transforming Randomized PCA.....'
		transformedXs = rca_transform(X_load)
	else:
		print 'Transforming PCA.....'
		transformedXs = pca_transform(X_load)



	for i in range(len(transformedXs)):
		cum_score = cum_scores[i]
		print 'cum_score:',cum_score
		X_full = transformedXs[i]
		print cum_score
		best_metric,best_grid, best_estimator,best_predict = trainGrid(X_full,y_full,estimator,grid)
		saveNewResults(funcType, fpType, regType, best_metric,best_grid, best_estimator,best_predict,cum_score,pca)
	return

def trainGrid(X_full,y_full,estimator,grid):
	scores = []
	best_score = NEG
	best_grid = None
	best_predict = None
	best_estimator = None
	best_metric = [best_score,NEG,NEG,NEG]
	for g in ParameterGrid(grid):
		estimator.set_params(**g)
		print estimator
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
		print score
		scores += [score]

		if score >= best_score:
			best_score = score
			R = r_score(y_true,y_predict)
			MSE = mean_squared_error(y_true, y_predict)
			RMSE = root_mean_squared_error(y_true,y_predict)
			MAE = mean_absolute_error(y_true, y_predict)
			MAPE = mean_absolute_percentage_error(y_true, y_predict)
			best_grid = g
			best_estimator = estimator
			best_metric = [score, R, MSE, RMSE, MAE,MAPE]
			best_predict = y_predict

	return best_metric,best_grid, best_estimator,best_predict


def supervised_transform(X_train,estimator):
	features = estimator.feature_importances_
	std = np.std([tree.feature_importances_ for tree in estimator.estimators_],
                 axis=0)
	indices = np.argsort(features)[::-1]
	reducedIndices = indices[:len(indices)-20]
	X_train_reduced = []
	for row in X_train:
		row_reduced = [row[i] for i in reducedIndices]
		X_train_reduced += [row_reduced]

	print 'reducing features'

	return array(X_train_reduced)

def runSupervisedGrid(funcType,fpType="AtomPair",regType="randomForest"):

	funcType = 'B3LYP'
	regType = 'randomForest'
	print fpType
	for idx in range(9,21):
		print
		print 'Step',idx+1,'of the supervised feature selection'
		print '_________________________________________'
		print
		# if 'selected' in fpType:
		# 	X_full = StandardScaler().fit_transform(loadData("X_full_"+fpType.replace('selected','')+str(idx),"FP"))
		# else:
		X_full = loadData("X_full_"+fpType+str(idx),"FP")
		y_true = array(loadData("y_full_"+funcType,"FP"))

		grid = paramsGrid[regType]
		estimator = getEstimator(regType)
		best_metric,best_grid, best_estimator,best_predict = trainGrid(X_full,y_true,estimator,grid)
		if idx<9:
			X_full_reduced = supervised_transform(X_full, best_estimator)
			print len(X_full_reduced),len(X_full_reduced[0])
			saveData(X_full_reduced,'X_full_'+fpType+str(idx+1))
		saveNewResults(funcType, fpType, regType, best_metric,best_grid, best_estimator,best_predict,idx,'supervised')

	return


def main():
	funcType = argv[1]
	fpType = argv[2]
	regType = argv[3]
	if len(argv)>4:
		pcaType = argv[4]
		if 'sfs' in pcaType:
			runSupervisedGrid(funcType,fpType,regType)
		else:
			GridSearchPCA(funcType,fpType,regType,pcaType)
	else:
		GridSearchNoCV(funcType,fpType,regType,True)

if __name__ == '__main__':
    main()
