class Model_Generator():#(TS_Helper):
# def load_file_list(path:str,global_tasks:int,smiles_col:str,task_start:int,local_num_tasks:int,wich_split:str,fold:int):
    """
    Loads the x_data,y_data,smiles for (Train or Test or Validation) and  returns a generator with the data, to access the data use next(generator), or a for loop
    
    Data is returned in the following order: x_data, y_data, smiles
    
    path: is the path to the folder where the files are located, use relative path
    global_tasks: is the number of tasks you want to iterate over, considering that each task is in a different file
    smiles_col: is the name of the column that contains the smiles
    task_start: is the column where the tasks start
    local_num_tasks: is the number of tasks in each file
    wich_split: is the type of split you want to load, train, val or test, 
    FOLD: is the fold you want to load, considering that FOLD is in the name of the file
    """
    data = [path+"/"+train for train in os.listdir(path) if train.find(wich_split)!=-1 and train.   (str(fold))!=-1 and train.endswith(".csv")]
    for task in range(global_tasks):
        yield commons.load_dataset(data[task],smiles_col,task_start,local_num_tasks)

def assign_fingerprints(smiles:list[str],fp_size:int,radius:int,feat:bool):
    """
    returns fingerprints for the given smiles
    smiles: is a list of smiles
    fp_size: is the size of the bit string
    radius: is the radius of the circular fingerprint
    feat: is a boolean that indicates if you want to consider pharmacophoric features
    """
    return commons.assing_fp(smiles,fp_size,radius,feat)

def assignXY(fp_size:int,radius:int,feat:bool,path:str,global_tasks:int,smiles_col:str,task_start:int,local_num_tasks:int,wich_split:str,fold:int):
    """
    returns a generator with the {x_data,y_data} of the given split (train, val or test) to access the data use next(generator), or a for loop
    - x_data (Fingerprint values),
    - y_data (nd.array with integer values),
    fp_size: is the size of the bit string
    radius: is the radius of the circular fingerprint
    feat: is a boolean that indicates if you want to consider pharmacophoric features
    path: is the path to the folder where the files are located, use relative path
    tasks: is the number of tasks you want to iterate over, considering that each task is in a different file
    Wich_split: is the type of split you want to load, train, val or test,
    FOLD: is the fold you want to load, considering that FOLD is in the name of the file
    """
    for _,y_data,smiles in load_file_list(path,global_tasks,smiles_col,task_start,local_num_tasks,wich_split,fold):
        y_data = y_data.ravel()
        y_data = np.array(y_data).astype(int)
        yield assign_fingerprints(smiles,fp_size,radius,feat),y_data
        
def build_model(X_data,y_data,selected_model,use_default_params:bool=True):
    """
    returns a model with the best parameters found using bayesian optimization
    X_data: is the x_data
    y_data: is the y_data
    """
    build_model = model_generator.Models[SELECTED_MODEL]
    CLASSIFIER = build_model["classifier"]
    PARAMS = build_model["params"]
    if use_default_params:
        lgbm_best_params = {'learning_rate': 0.04690935629679825, 'max_depth': 8, 'n_estimators': 47, 'num_leaves': 5, 'subsample': 0.23402958965378692}
        rf_best_params = {'max_depth': 5, 'max_features': 'sqrt', 'n_estimators': 140}
        svm_best_params = {'C': 0.0410104548749355, 'degree': 6, 'kernel': 'sigmoid'}
    else:
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
        scorer = make_scorer(mean_squared_error)
        best_model = BayesSearchCV(CLASSIFIER,
            PARAMS,
            n_iter=1, # Number of parameter settings that are sampled
            cv=cv,
            scoring = scorer,
            refit = True, # Refit the best estimator with the entire dataset.
            random_state=42,
            n_jobs = -1
        )
        best_model.fit(X_data, y_data)
        best_params = best_model.best_params_
    
    #setModelParams = lambda selected_model,use_default_params: rf_best_params if selected_model == model_generator.RF and use_default_params else best_params or lgbm_best_params if selected_model == model_generator.LGBM and use_default_params else best_params or svm_best_params if selected_model == model_generator.SVM and use_default_params else best_params
    if selected_model == model_generator.RF and use_default_params:
            params = rf_best_params
    elif selected_model == model_generator.LGBM and use_default_params:
            params = lgbm_best_params
    elif selected_model == model_generator.SVM and use_default_params:
            params = svm_best_params
    elif not use_default_params:
            params = best_params
    if selected_model not in model_generator.Models.keys():
           raise "Not implemented yet"
    model = CLASSIFIER.set_params(**params)
    model.fit(X_data,y_data)
    return model

def buildThenFitThenGetStats(fp_size:int,radius:int,feat:bool,path:str,global_tasks:int,smiles_col:str,task_start:int,local_num_tasks:int,which_splits:list[str],fold,selected_model,useDefaultParams:bool=True)->tuple[pd.DataFrame,str]:
    """
    Returns a generator with the stats in a Dataframe of the model for each given split (train, val or test)
    
    To access the data use next(generator), or a for loop
    Data is returned in the following order: dataframe:pd.Dataframe, split:str
    
    fp_size: is the size of the bit string
    
    radius: is the radius of the circular fingerprint
    
    feat: is a boolean that indicates if you want to consider pharmacophoric features
    
    path: is the path to the folder where the files are located, use relative path
    
    global_tasks: is the number of tasks you want to iterate over, considering that each task is in a different file
    
    smiles_col: is the name of the column that contains the smiles
    
    task_start: is the column where the tasks start
    
    local_num_tasks: is the number of tasks in each file
    
    wich_split: is the type of split you want to load, train, val or test, 
    
    fold: is the fold you want to load, considering that FOLD is in the name of the file
    
    useDefaultParams: is a boolean that indicates if you want to use the default parameters or the best parameters found using bayesian optimization
    """
    """for split in which_splits:
         for x_data,y_data in assignXY(fp_size,radius,feat,path,global_tasks,smiles_col,task_start,local_num_tasks,split,fold):
            model = build_model(x_data,y_data,selected_model,useDefaultParams)
            text,df = ml_helper.get_ML_StatsForNSplits(model,X_train=x_data,y_train=y_data)
            print(text)
            df["split"] = split
            split_type = "Random"
            df["Model_type"] = selected_model
            df["Tasks"] = str(global_tasks)
            yield df,split"""

        
# Models = {
#         "Dense":T.keras.models.Sequential([
#             T.keras.layers.Dense(64,activation="relu"),
#             T.keras.layers.Dense(64,activation="relu"),
#             T.keras.layers.Dense(1,activation="sigmoid")
#         ]),
#         "LSTM":T.keras.models.Sequential([
#             T.keras.layers.LSTM(64,activation="relu",return_sequences=True),
#             T.keras.layers.LSTM(64,activation="relu"),
#             T.keras.layers.Dense(1,activation="sigmoid")
#         ]),
#         "GRU":T.keras.models.Sequential([
#             T.keras.layers.GRU(64,activation="relu",return_sequences=True),
#             T.keras.layers.GRU(64,activation="relu"),
#             T.keras.layers.Dense(1,activation="sigmoid")
#         ]),
#         "Conv1D":T.keras.models.Sequential([
#             T.keras.layers.Conv1D(64,3,activation="relu"),
#             T.keras.layers.MaxPool1D(3),
#             T.keras.layers.Conv1D(64,3,activation="relu"),
#             T.keras.layers.MaxPool1D(3),
#             T.keras.layers.Flatten(),
#             T.keras.layers.Dense(1,activation="sigmoid")
#         ]),
#         "Conv2D":T.keras.models.Sequential([
#             T.keras.layers.Conv2D(64,3,activation="relu"),
#             T.keras.layers.MaxPool2D(3),
#             T.keras.layers.Conv2D(64,3,activation="relu"),
#             T.keras.layers.MaxPool2D(3),
#             T.keras.layers.Flatten(),
#             T.keras.layers.Dense(1,activation="sigmoid")
#         ]),
#         "Conv3D":T.keras.models.Sequential([
#             T.keras.layers.Conv3D(64,3,activation="relu"),
#             T.keras.layers.MaxPool3D(3),
#             T.keras.layers.Conv3D(64,3,activation="relu"),
#             T.keras.layers.MaxPool3D(3),
#             T.keras.layers.Flatten(),
#             T.keras.layers.Dense(1,activation="sigmoid")
#         ]),
#         "RNN":T.keras.models.Sequential([
#             T.keras.layers.SimpleRNN(64,activation="relu",return_sequences=True),
#             T.keras.layers.SimpleRNN(64,activation="relu"),
#             T.keras.layers.Dense(1,activation="sigmoid")
#         ]),
#         "Bidirectional":T.keras.models.Sequential([
#             T.keras.layers.Bidirectional(T.keras.layers.LSTM(64,activation="relu",return_sequences=True)),
#             T.keras.layers.Bidirectional(T.keras.layers.LSTM(64,activation="relu")),
#             T.keras.layers.Dense(1,activation="sigmoid")
#         ]),
#         "TimeDistributed":T.keras.models.Sequential([
#             T.keras.layers.TimeDistributed(T.keras.layers.Dense(64,activation="relu")),
#             T.keras.layers.TimeDistributed(T.keras.layers.Dense(64,activation="relu")),
#             T.keras.layers.Dense(1,activation="sigmoid")
#         ]),
#         "Attention":T.keras.models.Sequential([
#             T.keras.layers.Attention(),
#             T.keras.layers.Dense(1,activation="sigmoid")
#         ]),
#         "Transformer":T.keras.models.Sequential([
#             T.keras.layers.Transformer(),
#             T.keras.layers.Dense(1,activation="sigmoid")
#         ]),
#         "TransformerEncoder":T.keras.models.Sequential([
#             T.keras.layers.TransformerEncoder(),
#             T.keras.layers.Dense(1,activation="sigmoid")
#         ]),
#         "TransformerDecoder":T.keras.models.Sequential([
#             T.keras.layers.TransformerDecoder(),
#             T.keras.layers.Dense(1,activation="sigmoid")
#         ]),
#         "TransformerEncoderLayer":T.keras.models.Sequential([
#             T.keras.layers.TransformerEncoderLayer(),
#             T.keras.layers.Dense(1,activation="sigmoid")
#         ]),
#         "TransformerDecoderLayer":T.keras.models.Sequential([
#             T.keras.layers.TransformerDecoderLayer(),
#             T.keras.layers.Dense(1,activation="sigmoid")
#         ]),
#         "TransformerEncoderBlock":T.keras.models.Sequential([
#             T.keras.layers.TransformerEncoderBlock(),
#             T.keras.layers.Dense(1,activation="sigmoid")
#         ]),
#         "TransformerDecoderBlock":T.keras.models.Sequential([
#             T.keras.layers.TransformerDecoderBlock(),
#             T.keras.layers.Dense(1,activation="sigmoid")
#         ]),
#         "TransformerEncoderStack":T.keras.models.Sequential([
#             T.keras.layers.TransformerEncoderStack(),
#             T.keras.layers.Dense(1,activation="sigmoid")
#         ]),
#         "TransformerDecoderStack":T.keras.models.Sequential([
#             T.keras.layers.TransformerDecoderStack(),
#             T.keras.layers.Dense(1,activation="sigmoid")
#         ]),
#         "TransformerEncoderHead":T.keras.models.Sequential([
#             T.keras.layers.TransformerEncoderHead(),
#             T.keras.layers.Dense(1,activation="sigmoid")
#         ]),
#         "TransformerDecoderHead":T.keras.models.Sequential([
#             T.keras.layers.TransformerDecoderHead(),
#             T.keras.layers.Dense(1,activation="sigmoid")
#         ]),
#     }
        def __init__(self,):
                super().__init__()

def eufrasiaMLBestModels(SELECTED_MODEL,CLASSIFIER,rf_best_params,lgbm_best_params,svm_best_params,model_generator):

        if SELECTED_MODEL == model_generator.RF:
                model = CLASSIFIER.set_params(**rf_best_params)
        #model = best_model.best_estimator_
        if SELECTED_MODEL == model_generator.LGBM:
                model = CLASSIFIER.set_params(**lgbm_best_params)
        #model = best_model.best_estimator_

        if SELECTED_MODEL == model_generator.SVM:
                model = CLASSIFIER.set_params(**svm_best_params)
                #model = best_model.best_estimator_
        else:
                lgbm_best_params = {'learning_rate': 0.04690935629679825, 'max_depth': 8, 'n_estimators': 47, 'num_leaves': 5, 'subsample': 0.23402958965378692}
                rf_best_params = {'max_depth': 5, 'max_features': 'sqrt', 'n_estimators': 140}
                svm_best_params = {'C': 0.0410104548749355, 'degree': 6, 'kernel': 'sigmoid'}
          
def get_RegressionStatsFor_Train_Test_Validation(self,model,X_train,y_train,X_test,y_test,X_val,y_val,task:int) :#-> Tuple[str,tuple[float,float,float]]:
        regression_prediction_train = pd.DataFrame(data={"pred":model.predict(X_train)[:,task],"y":y_train[:,task]},index=range(len(y_train)))
        regression_prediction_val = pd.DataFrame(data={"pred":model.predict(X_val)[:,task],"y":y_val[:,task]},index=range(len(y_val)))
        regression_prediction_test = pd.DataFrame(data={"pred":model.predict(X_test)[:,task],"y":y_test[:,task]},index=range(len(y_test)))
        
        train_val_test = [regression_prediction_train,regression_prediction_val,regression_prediction_test]
        
        train_text = f"For Training in task {task} \n "
        train_statistics,_ = self.calc_RegressionStatistics(regression_prediction_train)
        train_text += train_statistics

        val_text = f"For Validation in task {task} \n "
        val_statistics,_ = self.calc_RegressionStatistics(regression_prediction_val)
        val_text += val_statistics

        test_text = f"For Testing in task {task} \n "
        test_statistics,_ = self.calc_RegressionStatistics(regression_prediction_test)
        test_text += test_statistics

        return train_text+"\n"+val_text+"\n"+test_text,train_val_test
    
def get_ClassificationStatsFor_Train_Test_Validation(self,model,X_train,y_train,X_test,y_test,X_val,y_val,task:int,threshold:float): #-> Tuple[str, tuple[float,float,float],Tuple[np.ndarray],Tuple[np.ndarray],Tuple[np.ndarray]]:
        classification_prediction_train = self.set_prediction_threshold(model, X_train, y_train, task, threshold)
        classification_prediction_val = self.set_prediction_threshold(model, X_val, y_val, task, threshold)
        classification_prediction_test = self.set_prediction_threshold(model, X_test, y_test, task, threshold)
        
        train_confusion_matrix, (TP, TN, FP, FN) = self.calc_confusion_matrix(classification_prediction_train)
        val_confusion_matrix, (TP_val, TN_val, FP_val, FN_val) = self.calc_confusion_matrix(classification_prediction_val)
        test_confusion_matrix, (TP_test, TN_test, FP_test, FN_test) = self.calc_confusion_matrix(classification_prediction_test)

        train_val_test = [classification_prediction_train,classification_prediction_val,classification_prediction_test]
        
        train_text = f"For Training in task {task} \n "
        train_statistics,_ = self.calc_Statistics(TP, TN, FP, FN, classification_prediction_train)
        train_text += train_statistics

        val_text = f"For Validation in task {task} \n "
        val_statistics,_ = self.calc_Statistics(TP_val, TN_val, FP_val, FN_val, classification_prediction_val)
        val_text += val_statistics
        
        test_text = f"For Testing in task {task} \n "
        test_statistics,_ = self.calc_Statistics(TP_test, TN_test, FP_test, FN_test, classification_prediction_test)
        test_text += test_statistics

        return train_text+"\n"+val_text+"\n"+test_text,train_val_test,train_confusion_matrix,val_confusion_matrix,test_confusion_matrix

def seiLa(train_dataset, val_dataset, test_dataset, y_train, y_val, y_test, model):
        # #Statistical characteristico of model without 3-sigma rule

        # prediction_train = model.predict(train_dataset)
        # prediction_val = model.predict(val_dataset)
        # prediction_test = model.predict(test_dataset)


        # for index1 in range(prediction_train.shape[1]):

        #         train_pred = pd.DataFrame(y_train[:,index1],prediction_train[:,index1]) 
        #         train_pred['y_pred'] = train_pred.index
        #         train_pred = train_pred.rename(columns = {0: 'y_obs'})
        #         train_pred2 = train_pred.dropna()
        #         train_pred2 = train_pred2.reset_index(drop=True)
        #         train_pred2['Folds'] = 'Train'
        #         train_pred2 = train_pred2.assign(Folds_error = abs(train_pred2['y_pred'] - train_pred2['y_obs']))
        #         train_pred2['Folds error Mean'] = train_pred2['Folds_error'].mean() 
        #         train_pred2['Folds error 3*sigma'] = train_pred2['Folds_error'].std()
        #         train_pred2['Folds error 3*sigma'] = train_pred2['Folds error 3*sigma']*3

        #         for index2 in range(prediction_val.shape[1]):
                
        #                 val_pred = pd.DataFrame(y_val[:,index2],prediction_val[:,index2])
        #                 val_pred['y_pred'] = val_pred.index
        #                 val_pred = val_pred.rename(columns = {0: 'y_obs'})
        #                 val_pred2 = val_pred.dropna()
        #                 val_pred2 = val_pred2.reset_index(drop=True)
        #                 val_pred2['Folds'] = 'val'
        #                 val_pred2 = val_pred2.assign(Folds_error = abs(val_pred2['y_pred'] - val_pred2['y_obs']))
        #                 val_pred2['Folds error Mean'] = val_pred2['Folds_error'].mean() 
        #                 val_pred2['Folds error 3*sigma'] = val_pred2['Folds_error'].std()
        #                 val_pred2['Folds error 3*sigma'] = val_pred2['Folds error 3*sigma']*3
                

        #                 for index3 in range(prediction_test.shape[1]):
                        
        #                         test_pred = pd.DataFrame(y_test[:,index3],prediction_test[:,index3])
        #                         test_pred['y_pred'] = test_pred.index
        #                         test_pred = test_pred.rename(columns = {0: 'y_obs'})
        #                         test_pred2 = test_pred.dropna()
        #                         test_pred2 = test_pred2.reset_index(drop=True)
        #                         test_pred2['Folds'] = 'Test'
        #                         test_pred2 = test_pred2.assign(Folds_error = abs(test_pred2['y_pred'] - test_pred2['y_obs']))
        #                         test_pred2['Folds error Mean'] = test_pred2['Folds_error'].mean() 
        #                         test_pred2['Folds error 3*sigma'] = test_pred2['Folds_error'].std()
        #                         test_pred2['Folds error 3*sigma'] = test_pred2['Folds error 3*sigma']*3

        #         crossval_df = pd.concat([train_pred2, val_pred2, test_pred2], axis=0).reset_index(drop=True)


        #         if index1 == index2 and index1 == index3:
                        
        #                 r2  = (train_pred2["y_obs"].corr(train_pred2["y_pred"]))    
        #                 print(("Results for task {} (train)").format(index2+1))
        #                 print("r^2\t%.2f" % r2)
        #                 print ("rmse\t%.2f" % sqrt(mean_squared_error(train_pred2["y_obs"],train_pred2["y_pred"])))
        #                 print ("mse\t%.2f" % (mean_squared_error(train_pred2["y_obs"],train_pred2["y_pred"])))
        #                 print ("mae\t%.2f"  %mean_absolute_error(train_pred2["y_obs"],train_pred2["y_pred"]))   

        #                 r2 = (val_pred2["y_obs"].corr(val_pred2["y_pred"]))
        #                 print(("Results for task {} (validation)").format(index3+1))
        #                 print("r^2\t%.2f" % r2)
        #                 print ("rmse\t%.2f"  % sqrt(mean_squared_error(val_pred2["y_pred"],val_pred2["y_obs"])))
        #                 print ("mse\t%.2f"  % (mean_squared_error(val_pred2["y_pred"],val_pred2["y_obs"])))
        #                 print ("mae\t%.2f"  % mean_absolute_error(val_pred2["y_pred"],val_pred2["y_obs"]))
                        
        #                 r2 = (test_pred2["y_obs"].corr(test_pred2["y_pred"])) 
        #                 print(("Results for task {} (test)").format(index1+1))
        #                 print("r^2\t%.2f" % r2)
        #                 print ("rmse\t%.2f"  % sqrt(mean_squared_error(test_pred2["y_pred"],test_pred2["y_obs"]))) 
        #                 print ("mse\t%.2f"  % (mean_squared_error(test_pred2["y_pred"],test_pred2["y_obs"])))
        #                 print ("mae\t%.2f"  % mean_absolute_error(test_pred2["y_pred"],test_pred2["y_obs"]))

        #                 g = sns.lmplot(x="y_pred", y="y_obs", hue="Folds", data=crossval_df, fit_reg=False, height=7, 
        #                 markers=["o", "o", "o"], palette="rocket",scatter_kws={"s": 50,'alpha':0.9},  aspect=30/30)
        #                 sns.regplot(x="y_pred", y="y_obs", data=crossval_df, scatter=False, ax=g.axes[0, 0]) 
        pass

def notes():        
        
        # for index1 in range(prediction_train.shape[1]):

        #     train_pred = pd.DataFrame(y_train[:,index1],prediction_train[:,index1]) 
        #     train_pred['y_pred'] = train_pred.index
        #     train_pred = train_pred.rename(columns = {0: 'y_obs'})
        #     train_pred2 = train_pred.dropna()
        #     train_pred2 = train_pred2.reset_index(drop=True)
        #     train_pred2['Folds'] = 'Train'
        #     train_pred2 = train_pred2.assign(Folds_error = abs(train_pred2['y_pred'] - train_pred2['y_obs']))
        #     train_pred2['Folds error Mean'] = train_pred2['Folds_error'].mean() 
        #     train_pred2['Folds error 3*sigma'] = train_pred2['Folds_error'].std()
        #     train_pred2['Folds error 3*sigma'] = train_pred2['Folds error 3*sigma']*3
        #     train_pred2=train_pred2[train_pred2['Folds_error']<=(train_pred2['Folds error 3*sigma'])] #keep only the ones that are within +3 to -3 standard deviations in the column 'Data'.

        #     for index2 in range(prediction_val.shape[1]):
                
        #         val_pred = pd.DataFrame(y_val[:,index2],prediction_val[:,index2])
        #         val_pred['y_pred'] = val_pred.index
        #         val_pred = val_pred.rename(columns = {0: 'y_obs'})
        #         val_pred2 = val_pred.dropna()
        #         val_pred2 = val_pred2.reset_index(drop=True)
        #         val_pred2['Folds'] = 'val'
        #         val_pred2 = val_pred2.assign(Folds_error = abs(val_pred2['y_pred'] - val_pred2['y_obs']))
        #         val_pred2['Folds error Mean'] = val_pred2['Folds_error'].mean() 
        #         val_pred2['Folds error 3*sigma'] = val_pred2['Folds_error'].std()
        #         val_pred2['Folds error 3*sigma'] = val_pred2['Folds error 3*sigma']*3
        #         val_pred2=val_pred2[val_pred2['Folds_error']<=(val_pred2['Folds error 3*sigma'])]#keep only the ones that are within +3 to -3 standard deviations in the column 'Data'.
        
        #         for index3 in range(prediction_test.shape[1]):

        #             test_pred = pd.DataFrame(y_test[:,index3],prediction_test[:,index3])
        #             test_pred['y_pred'] = test_pred.index
        #             test_pred = test_pred.rename(columns = {0: 'y_obs'})
        #             test_pred2 = test_pred.dropna()
        #             test_pred2 = test_pred2.reset_index(drop=True)
        #             test_pred2['Folds'] = 'Test'
        #             test_pred2 = test_pred2.assign(Folds_error = abs(test_pred2['y_pred'] - test_pred2['y_obs']))
        #             test_pred2['Folds error Mean'] = test_pred2['Folds_error'].mean() 
        #             test_pred2['Folds error 3*sigma'] = test_pred2['Folds_error'].std()
        #             test_pred2['Folds error 3*sigma'] = test_pred2['Folds error 3*sigma']*3
        #             test_pred2=test_pred2[test_pred2['Folds_error']<=(test_pred2['Folds error 3*sigma'])] #keep only the ones that are within +3 to -3 standard deviations in the column 'Data'.

        #             crossval_df = pd.concat([train_pred2, val_pred2, test_pred2], axis=0).reset_index(drop=True)

        #             if index1 == index2 and index1 == index3:
                                
        #                 r2 = (train_pred2["y_obs"].corr(train_pred2["y_pred"]))    
        #                 print(("Results for task {} (train)").format(index2+1))
        #                 print("r^2\t%.2f" % r2)
        #                 print ("rmse\t%.2f" % sqrt(mean_squared_error(train_pred2["y_obs"],train_pred2["y_pred"])))
        #                 print ("mse\t%.2f" % (mean_squared_error(train_pred2["y_obs"],train_pred2["y_pred"])))
        #                 print ("mae\t%.2f"  %mean_absolute_error(train_pred2["y_obs"],train_pred2["y_pred"]))   

        #                 r2= (val_pred2["y_obs"].corr(val_pred2["y_pred"]))
        #                 print(("Results for task {} (validation)").format(index3+1))
        #                 print("r^2\t%.2f" % r2)
        #                 print ("rmse\t%.2f"  % sqrt(mean_squared_error(val_pred2["y_pred"],val_pred2["y_obs"])))
        #                 print ("mse\t%.2f"  % (mean_squared_error(val_pred2["y_pred"],val_pred2["y_obs"])))
        #                 print ("mae\t%.2f"  % mean_absolute_error(val_pred2["y_pred"],val_pred2["y_obs"]))
                        
        #                 r2 = (test_pred2["y_obs"].corr(test_pred2["y_pred"])) 
        #                 print(("Results for task {} (test)").format(index1+1))
        #                 print("r^2\t%.2f" % r2)
        #                 print ("rmse\t%.2f"  % sqrt(mean_squared_error(test_pred2["y_pred"],test_pred2["y_obs"]))) 
        #                 print ("mse\t%.2f"  % (mean_squared_error(test_pred2["y_pred"],test_pred2["y_obs"])))
        #                 print ("mae\t%.2f"  % mean_absolute_error(test_pred2["y_pred"],test_pred2["y_obs"]))

        #                 g = sns.lmplot(x="y_pred", y="y_obs", hue="Folds", data=crossval_df, fit_reg=False, height=7, 
        #                 markers=["o", "o", "o"], palette="rocket",scatter_kws={"s": 50,'alpha':0.9},  aspect=30/30)
        #                 sns.regplot(x="y_pred", y="y_obs", data=crossval_df, scatter=False, ax=g.axes[0, 0])  
        pass

def unused_code(self):
        unused_13 = """                    # print("Before 3 Sigma:\n",full_text,end="\n\n")
                    # y_pred = Commons.ndarrayTo1Darray(Commons,model.predict(X_data)[:,task])
                    # y_data = Commons.ndarrayTo1Darray(Commons,y_data)
                    # y_pred,y_obs = self.set_3SigmaStats(y_data,y_pred)
                    # full_text,_ = self.calc_RegressionStatistics(pred_y=y_pred,obs_y=y_obs)
                    # print("After 3 Sigma:\n",full_text,end="\n\n")"""
        unused_12 = """# for task in range(tasks):
            
            # if self.model_type == self.Classification:
            #     full_text,cross_val,train_confusion_matrix,val_confusion_matrix,test_confusion_matrix = self.get_ClassificationStatsForAllSplits(model,X_train,y_train,X_test,y_test,X_val,y_val,task,threshold)
            #     print(full_text)
            #     self.plot_Classification(train_confusion_matrix,["TP","FP"],f"Confusion Matrix for Training in Task {task}")
            #     self.plot_Classification(val_confusion_matrix,["TP","FP"],f"Confusion Matrix for Validation in Task {task}")
            #     self.plot_Classification(test_confusion_matrix,["TP","FP"],f"Confusion Matrix for Testing in Task {task}")

            # if self.model_type == self.Regression:
            #     full_text,cross_val = self.get_RegressionStatsForAllSplits(model,X_train,y_train,X_test,y_test,X_val,y_val,task)
            #     print(full_text)
            #     self.plot_Regression(cross_val[0],cross_val[1],cross_val[2])              
"""
        unused_11="""for index1 in range(prediction_train.shape[1]):

#     train_pred = pd.DataFrame(y_train[:,index1],prediction_train[:,index1]) 
#     train_pred['y_pred'] = train_pred.index
#     train_pred = train_pred.rename(columns = {0: 'y_obs'})
#     train_pred2 = train_pred.dropna()
#     train_pred2 = train_pred2.reset_index(drop=True)
#     train_pred2['Folds'] = 'Train'
#     train_pred2 = train_pred2.assign(Folds_error = abs(train_pred2['y_pred'] - train_pred2['y_obs']))
#     train_pred2['Folds error Mean'] = train_pred2['Folds_error'].mean() 
#     train_pred2['Folds error 3*sigma'] = train_pred2['Folds_error'].std()
#     train_pred2['Folds error 3*sigma'] = train_pred2['Folds error 3*sigma']*3

#     for index2 in range(prediction_val.shape[1]):
         
#         val_pred = pd.DataFrame(y_val[:,index2],prediction_val[:,index2])
#         val_pred['y_pred'] = val_pred.index
#         val_pred = val_pred.rename(columns = {0: 'y_obs'})
#         val_pred2 = val_pred.dropna()
#         val_pred2 = val_pred2.reset_index(drop=True)
#         val_pred2['Folds'] = 'val'
#         val_pred2 = val_pred2.assign(Folds_error = abs(val_pred2['y_pred'] - val_pred2['y_obs']))
#         val_pred2['Folds error Mean'] = val_pred2['Folds_error'].mean() 
#         val_pred2['Folds error 3*sigma'] = val_pred2['Folds_error'].std()
#         val_pred2['Folds error 3*sigma'] = val_pred2['Folds error 3*sigma']*3
          
#         for index3 in range(prediction_test.shape[1]):
         
#             test_pred = pd.DataFrame(y_test[:,index3],prediction_test[:,index3])
#             test_pred['y_pred'] = test_pred.index
#             test_pred = test_pred.rename(columns = {0: 'y_obs'})
#             test_pred2 = test_pred.dropna()
#             test_pred2 = test_pred2.reset_index(drop=True)
#             test_pred2['Folds'] = 'Test'
#             test_pred2 = test_pred2.assign(Folds_error = abs(test_pred2['y_pred'] - test_pred2['y_obs']))
#             test_pred2['Folds error Mean'] = test_pred2['Folds_error'].mean() 
#             test_pred2['Folds error 3*sigma'] = test_pred2['Folds_error'].std()
#             test_pred2['Folds error 3*sigma'] = test_pred2['Folds error 3*sigma']*3

            crossval_df = pd.concat([train_pred2, val_pred2, test_pred2], axis=0).reset_index(drop=True)

            if index1 == index2 and index1 == index3:
                    
                r2  = (train_pred2["y_obs"].corr(train_pred2["y_pred"]))    
                print(("Results for task {} (train)").format(index2+1))
                print("r^2\t%.2f" % r2)
                print ("rmse\t%.2f" % sqrt(mean_squared_error(train_pred2["y_obs"],train_pred2["y_pred"])))
                print ("mse\t%.2f" % (mean_squared_error(train_pred2["y_obs"],train_pred2["y_pred"])))
                print ("mae\t%.2f"  %mean_absolute_error(train_pred2["y_obs"],train_pred2["y_pred"]))   

                r2 = (val_pred2["y_obs"].corr(val_pred2["y_pred"]))
                print(("Results for task {} (validation)").format(index3+1))
                print("r^2\t%.2f" % r2)
                print ("rmse\t%.2f"  % sqrt(mean_squared_error(val_pred2["y_pred"],val_pred2["y_obs"])))
                print ("mse\t%.2f"  % (mean_squared_error(val_pred2["y_pred"],val_pred2["y_obs"])))
                print ("mae\t%.2f"  % mean_absolute_error(val_pred2["y_pred"],val_pred2["y_obs"]))
                
                r2 = (test_pred2["y_obs"].corr(test_pred2["y_pred"])) 
                print(("Results for task {} (test)").format(index1+1))
                print("r^2\t%.2f" % r2)
                print ("rmse\t%.2f"  % sqrt(mean_squared_error(test_pred2["y_pred"],test_pred2["y_obs"]))) 
                print ("mse\t%.2f"  % (mean_squared_error(test_pred2["y_pred"],test_pred2["y_obs"])))
                print ("mae\t%.2f"  % mean_absolute_error(test_pred2["y_pred"],test_pred2["y_obs"]))

                g = sns.lmplot(x="y_pred", y="y_obs", hue="Folds", data=crossval_df, fit_reg=False, height=7, 
                markers=["o", "o", "o"], palette="rocket",scatter_kws={"s": 50,'alpha':0.9},  aspect=30/30)
                sns.regplot(x="y_pred", y="y_obs", data=crossval_df, scatter=False, ax=g.axes[0, 0]) """
        unused_10 = """def calc_confusion_matrix(y_train:np.array,num_tasks:int,prediction_train:np.array,y_val:np.array,prediction_val:np.array):
    pass
    for task_idx in range(num_tasks):
        
        a = pd.DataFrame(y_train[:,task_idx],prediction_train[:,task_idx]) 
        a['y'] = a.index
        b = a.dropna()
        confusion = SK.confusion_matrix(b["y"], b[0])
        #[row, column]
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]

        
        for index2 in range(prediction_val.shape[1]):
            
            a_val = pd.DataFrame(y_val[:,index2],prediction_val[:,index2]) 
            a_val['y'] = a_val.index
            b_val = a_val.dropna()
            confusion_val = SK.confusion_matrix(b_val["y"], b_val[0])
            #[row, column]
            TP_val = confusion_val[1, 1]
            TN_val = confusion_val[0, 0]
            FP_val = confusion_val[0, 1]
            FN_val = confusion_val[1, 0]

            
            for index3 in range(prediction_test.shape[1]):
                
                a_test = pd.DataFrame(y_test[:,index3],prediction_test[:,index3]) 
                a_test['y'] = a_test.index
                b_test = a_test.dropna()
                confusion_test = SK.confusion_matrix(b_test["y"], b_test[0])
                #[row, column]
                TP_test = confusion_test[1, 1]
                TN_test = confusion_test[0, 0]
                FP_test = confusion_test[0, 1]
                FN_test = confusion_test[1, 0]

            
            if index1 == index2 == index3:
                
                print(("Results for task {} (training)").format(index1+1))
                print("ACC\t%.2f" % ((TN+TP)/(TN+TP+FN+FP)))
                print("MCC\t%.2f" % SK.matthews_corrcoef(b["y"], b[0]))
                print("kappa\t%.2f" % SK.cohen_kappa_score(b["y"], b[0]))
                print("SE\t%.2f" % (TP/(TP+FN)))
                print("SP\t%.2f" % (TN/(TN+FP)))
                print("PPV\t%.2f" % (TP/(TP+FP)))
                print("NPV\t%.2f" % (TN/(TN+FN)))
                print("TPR\t%.2f" %(TP/(TP+FN)))
                print("FPR\t%.2f" %(FP/(FP+TN)))
                print("F1\t%.2f" % SK.f1_score(b["y"], b[0]))
                
                print(("Results for task {} (validation)").format(index2+1))
                print("ACC\t%.2f" % ((TN_val+TP_val)/(TN_val+TP_val+FN_val+FP_val)))
                print("MCC\t%.2f" % SK.matthews_corrcoef(b_val["y"], b_val[0]))
                print("kappa\t%.2f" % SK.cohen_kappa_score(b_val["y"], b_val[0]))
                print("SE\t%.2f" % (TP_val/(TP_val+FN_val)))
                print("SP\t%.2f" % (TN_val/(TN_val+FP_val)))
                print("PPV\t%.2f" % (TP_val/(TP_val+FP_val)))
                print("NPV\t%.2f" % (TN_val/(TN_val+FN_val)))
                print("TPR\t%.2f" %(TP_val/(TP_val+FN_val)))
                print("FPR\t%.2f" %(FP_val/(FP_val+TN_val)))
                print("F1\t%.2f" % SK.f1_score(b_val["y"], b_val[0]))
                
                print(("Results for task {} (test)").format(index3+1))
                print("ACC\t%.2f" % ((TN_test+TP_test)/(TN_test+TP_test+FN_test+FP_test)))
                print("MCC\t%.2f" % SK.matthews_corrcoef(b_test["y"], b_test[0]))
                print("kappa\t%.2f" % SK.cohen_kappa_score(b_test["y"], b_test[0]))
                print("SE\t%.2f" % (TP_test/(TP_test+FN_test)))
                print("SP\t%.2f" % (TN_test/(TN_test+FP_test)))
                print("PPV\t%.2f" % (TP_test/(TP_test+FP_test)))
                print("NPV\t%.2f" % (TN_test/(TN_test+FN_test)))
                print("TPR\t%.2f" %(TP_test/(TP_test+FN_test)))
                print("FPR\t%.2f" %(FP_test/(FP_test+TN_test)))
                print("F1\t%.2f" % SK.f1_score(b_test["y"], b_test[0]))"""
        unused_9 = """
                # best_shap_bits = [commons.get_top_shap_values(X_f_hit,shap_value_single[i][0], 20) for i in range(3)]
                # best_shap_bits = [list(index_list[1]) for index_list in best_shap_bits]
                # for i,bits in enumerate(best_shap_bits):
                #     for j,bit in enumerate(bits):
                #        best_shap_bits[i][j] = int(bit.strip('bit-'))
                # best_shap_bits"""
        unused_1 = """new_dic =  {k:v for list_item in morgan_all[0] for (k,v) in list_item.items()}
                    sd = sorted(new_dic.items())
                    bit_info_im = {k:[v] for k,v in sd}
                    bit_info_im""" 
        unused_2 =  """range_bit = [] #save the list of bit idx 
                for n in (range(len(bi_all))):
                    range_int = list(bi_all[n])
                    range_bit.append(range_int)
                    range_bit"""
        unused_3 = """#Separate the atom and raio values  to the bitInfo data set

atom_value_all = []
raio_value_all = []

for k,n in enumerate(bit_info_im):

        first_value_int = [(bit_info_im[list(bit_info_im)[k]][0][0][0][0])] #atom position
        atom_value_all.append(first_value_int)

        second_value_int = [(bit_info_im[list(bit_info_im)[k]][0][0][0][1])] #raio position
        raio_value_all.append(second_value_int)"""
        unused_4 = """#get the smiles of the fragments"""
        unused_5= """#Separate the atom and raio values  to the bitInfo data set

atom_value = [[] for i in (range(len(bi_all)))]
radius_value = [[] for i in (range(len(bi_all)))]
print(len(mols))
for n in (range(len(bi_all))):

    for i in (range(len(bi_all[n]))):

        atom_index = (list((bi_all[n].values()))[i][0][0]) #atom position
        atom_value[n].append(atom_index)

        radius_index = (list((bi_all[n].values()))[i][0][1]) #raio position
        radius_value[n].append(radius_index)
 
"""
        unused_6 = """from IPython import display

#Get the smiles representation to the bit information 

def smilesbitfrag (mol_value, atom, raio):

     env = Chem.FindAtomEnvironmentOfRadiusN(mol_value, atom, raio)
     amap={}
     submol=Chem.PathToSubmol(mol_value,env,atomMap=amap)#bit info ecfp in Mol
     int_mol = Chem.MolToSmiles(submol)#

     return(int_mol)"""
        unused_7 = """fragment_moleculs = []
combined_list = []
print(len(mols))
for i, _ in enumerate(bi_all):

    fragment_moleculs_int = [smilesbitfrag(mols[i], radius_value[i][j], atom_value[i][j]) for j in (range(len(atom_value[i])))] 
    fragment_moleculs.append(fragment_moleculs_int)

for i, _ in enumerate(bi_all):   

    zip_list = (zip(range_bit[i], fragment_moleculs[i]))
    combined_list.append(dict(zip_list))

combined_list""" 
        unused_8 ="""def intersectiondic (self,dict_1:dict, dict_2:dict):

            symm = {k: dict_1.get(k, dict_2.get(k)) for k in dict_1.keys() ^ dict_2}
            
            inter = {k: dict_2[k] + dict_1[k] for k in dict_1.keys() & dict_2}
            
            sd = sorted(inter.items())
            sorted_d = {k:[v] for k,v in sd}

            unused_code =  morgan_all = {} #save the list of bit idx 
            return sorted_d
            comp = len(bi_all)-2
            some_it = 0

            while some_it < comp:

                for some_it, item in enumerate(bi_all):

                    it_2 = some_it + 1
                    
                    intem_values = intersectiondic(bi_all[some_it], bi_all[it_2])
                    morgan_all.setdefault(0, []).append(intem_values)
                    if it_2 > comp:

                        break
            morgan_all"""