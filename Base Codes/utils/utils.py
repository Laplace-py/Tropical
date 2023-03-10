import pandas as pd
import numpy as np
import math
import tensorflow as T
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
import itertools
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import SVG, Image
IPythonConsole.molSize = (400,400)
import matplotlib.pyplot as plt
import tabulate
import sklearn.metrics as SK
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering,OPTICS,AffinityPropagation,Birch
from typing import List, Set, Tuple, Union,Callable, Optional, Dict, Any
from dataclasses import dataclass
import os
from sklearn.model_selection import *
from sklearn.metrics import *
import PIL
from skopt import BayesSearchCV

@dataclass
class ChemType():
    """
    Typing only, no implementation.
    """
    def __init__(self):
        pass
@dataclass
class PillType():
    """
    Typing only, no implementation.
    """
    def __init__(self):
        pass
@dataclass
class GLOBALS():
    
    CLASSIFICATION: int = 0
    REGRESSION: int = 1
    SCAFFOLD_SPLIT: int = 0
    RANDOM_SPLIT: int = 1
    AVAILABLE_MODELS = [
        'LGBM','RF','SVM','KMeans',
        'GaussianMixture','DBSCAN',
        'OPTICS','AgglomerativeClustering',
        'AffinityPropagation','Birch'
        ]
    CLASSIFIER = "classifier"
    PARAMS = 'params'

class Commons():
    def __init__(self):
        pass
    
    def get_percentileAsFraction(self,total:int,percentile:float) -> str:
        return str(f"{math.floor(total*percentile)}"+"/"+f"{total}")
    
    def gen_PercentileAsFractionCol(self,df,df_TotalCol,df_PercentileCol) -> str:
        df["PercentileAsFraction"] = df.apply(lambda x: self.get_percentileAsFraction(x[df_TotalCol],x[df_PercentileCol]),axis=1)
        return df
    
    def strEqstr(self, a:str, b:str) -> bool:
        return a.lower() == b.lower() 

    def load_dataset(self, dataset:pd.DataFrame,smiles_col:str ,task_start:int=0, number_of_tasks:int=1) -> Tuple[pd.DataFrame,np.ndarray,list[str]]:
        task_end = task_start + number_of_tasks
        
        df = pd.read_csv(dataset, sep=",", low_memory=False)
        y_train = np.array(df.iloc[:,task_start:task_end].values)
        smiles = df[smiles_col].values
        print(f"Loaded dataset {dataset} with shape: {df.shape}")
        return df,y_train,smiles
    
    def load_dataset_list(self,path:str,how_many_files:int,smiles_col:str,task_start:int,local_num_tasks:int,wich_splits:List[str]="Train",fold:int=0):
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
        for wich_split in wich_splits:
            data = [path+"/"+train for train in os.listdir(path) if train.find(wich_split)!=-1 and train.find(str(fold))!=-1 and train.endswith(".csv")]
            for data_ in data:
                yield self.load_dataset(data_,smiles_col,task_start,local_num_tasks),wich_split
        
    def calc_fp(self,smiles:list,fp_size:int, radius:int, feat:bool) -> np.ndarray:
        """
        calcs morgan fingerprints as a numpy array.
        """
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        mol.UpdatePropertyCache(False)
        
        Chem.GetSSSR(mol)
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=fp_size, useFeatures=feat)
        a = np.zeros((0,), dtype=np.float32)
        a = Chem.DataStructs.ConvertToNumpyArray(fp, a)
        return np.asarray(fp)    
    
    def ndarrayTo1Darray(self,ndarray:np.ndarray) -> np.ndarray:
        return np.reshape(ndarray, (ndarray.shape[0],)) 
    
    def assing_fp(self,smiles:list,fp_size:int, radius:int, feat:bool) -> np.ndarray:
        canon_smiles = []
        for smile in smiles:
            #print(smile)
            try:
                cs = Chem.CanonSmiles(smile)
                canon_smiles.append(cs)
            except:
                #canon_smiles.append(smile)
                print(f"not valid smiles {smile}")
        #print(canon_smiles)
        #Make sure that the column where the smiles are is named SMILES
        descs = [self.calc_fp(smi, fp_size, radius,feat) for smi in canon_smiles]
        descs = np.asarray(descs, dtype=np.float32)
        return descs
    
    def setML_FPs(self,smiles,y_data:np.ndarray,size:int,radius:int,feat:bool) -> Tuple[np.ndarray,np.ndarray]:
        y_data = y_data.ravel()
        y_data = np.array(y_data).astype(int)
        return self.assing_fp(smiles,size,radius,feat),y_data

    def calc_similarity(self,smiles1:str, smiles2:str, fp_size:int, radius:int, feat:bool) -> float:
        """
        calcs morgan fingerprints as a numpy array.
        """
        mol1 = Chem.MolFromSmiles(smiles1, sanitize=True)
        mol1.UpdatePropertyCache(False)
        mol2 = Chem.MolFromSmiles(smiles2, sanitize=True)
        mol2.UpdatePropertyCache(False)
        Chem.GetSSSR(mol1)
        Chem.GetSSSR(mol2)
        fp1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol1, radius, nBits=fp_size, useFeatures=feat)
        fp2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol2, radius, nBits=fp_size, useFeatures=feat)
        return DataStructs.TanimotoSimilarity(fp1, fp2)

class Shap_Helper():
    def __init__(self):
        pass
    
    def get_bit_info(self, smiles:list,fp_size:int, radius:int, feat:bool)-> tuple[list[int],list[int],list[ChemType],list[int]]:
        """
        calcs morgan fingerprints as a numpy array.
        """
        bi_all = [] # set of bit morgan molecules
        mols = [] # set of molecules
        for i in range(len(smiles)):
            mol = Chem.MolFromSmiles(smiles[i])
            mols.append(mol)
            bi = {}
            _ = AllChem.GetMorganFingerprintAsBitVect(mol, nBits = fp_size ,radius = radius, bitInfo=bi)
            bi_all.append(bi)

        atom_value = [[] for _ in (range(len(bi_all)))]
        radius_value = [[] for _ in (range(len(bi_all)))]
        range_bit = []
        for n in (range(len(bi_all))):
            for i in (range(len(bi_all[n]))):

                atom_index = (list((bi_all[n].values()))[i][0][0]) #atom position
                atom_value[n].append(atom_index)

                radius_index = (list((bi_all[n].values()))[i][0][1]) #radius position
                radius_value[n].append(radius_index)
        for n in (range(len(bi_all))):
            range_int = list(bi_all[n])
            range_bit.append(range_int)
        return atom_value, radius_value, mols, range_bit
    
    def getSmilesFragFromBit (self,mol:Chem, atom:int, radius:int) -> str:

        env = Chem.FindAtomEnvironmentOfRadiusN(mol, atom, radius)
        amap={}
        submol=Chem.PathToSubmol(mol, env,atomMap=amap)#bit info ecfp in Mol
        fragment = Chem.MolToSmiles(submol)#

        return fragment
    
    def generateFragList(self,mols:list,radius_list:list,atom_list:list,bit_list:list) -> list[dict]:
        fragment_moleculs  = []
        combined_list = []
        
        for i in range(len(mols)):
            fragment_moleculs_int = [self.getSmilesFragFromBit(mols[i], radius_list[i][j], atom_list[i][j]) for j in (range(len(atom_list[i])))] 
            fragment_moleculs.append(fragment_moleculs_int)

        for i in range(len(mols)):   
            zip_list = (zip(bit_list[i], fragment_moleculs[i]))
            combined_list.append(dict(zip_list))
        return combined_list

    def formatDataforShapValues(self,fingerprint_array:np.array) -> pd.DataFrame:
        col_names = ['bit-{}'.format(_) for _ in range(fingerprint_array.shape[1])]
        df = pd.DataFrame(fingerprint_array, columns=col_names)
        return df

    def get_top_shap_values(self,shap_dataset:pd.DataFrame,shap_values, top_n = 10) -> tuple[list[float],list[str]]:
        top_indices = np.argsort(np.abs(shap_values))[-top_n:]
        top_shap_values = shap_values[top_indices]
        top_feature_names = shap_dataset.columns[top_indices]
        return top_shap_values, top_feature_names

    def get_bits_fromBestShaps(self,shap_dataset:pd.DataFrame,shap_values, top_n = 10) -> list[list[int]]:
        best_values,best_bits = [],[]
        for i in range(shap_dataset.shape[0]-1):
            values,bits = self.get_top_shap_values(shap_dataset,shap_values[i][0], top_n)
            bits = [int(bit.strip('bit-')) for bit in bits]
            best_values.append(values)
            best_bits.append(bits)
        return best_bits

    def get_highlightedMols(self,mols:ChemType,frag_list:list,mol_idx:list or int,b_bits:list) -> list[PillType]:

        highlights = {l:{k:[] for k in range(2048)} for l in range(len(frag_list)) }
        deletes = []
        images = []
        #drawer = rdMolDraw2D.MolDraw2DSVG(600,400)
        for i,frag in enumerate(frag_list):
            for bit in frag:
                highlights[i][bit] = list(mols[i].GetSubstructMatch(Chem.MolFromSmarts(frag[bit])))
            for high in highlights[i]:
                if highlights[i][high] == []:
                    deletes.append([i,high])

        for delete in deletes:
            if delete[1] not in b_bits[mol_idx]:
                del highlights[delete[0]][delete[1]]

        combination_of_bits = list(itertools.combinations(b_bits[mol_idx],2))
        for bits in combination_of_bits:
            if highlights[mol_idx][bits[0]] == [] or highlights[mol_idx][bits[1]] == []:
                continue
            else:
                highlight = highlights[mol_idx][bits[0]] + highlights[mol_idx][bits[1]]
                image = Chem.Draw.MolToImage(mols[mol_idx],highlightAtoms=highlight)
                images.append(image) 
                #drawer.DrawMolecule(mols[mol_idx],highlightAtoms=highlight)
            #highlight=list(mols[0].GetSubstructMatch(Chem.MolFromSmarts(combined_list[0][fragment_bit1]))) + list(mols[0].GetSubstructMatch(Chem.MolFromSmarts(combined_list[0][fragment_bit2])))
            return images

class Statistics():
    def __init__(self):
        pass
    Classification = GLOBALS.CLASSIFICATION
    Regression = GLOBALS.REGRESSION

    model_type: int = Classification or Regression
    assert model_type in [Classification, Regression], "model_type must be either Ts_Helper.Classification or Ts_Helper.Regression"
    
    def calc_confusion_matrix(self,prediction_df) -> Tuple[np.ndarray,Tuple[int,int,int,int]]:
        """
        prediction_df: dataframe with predictions and target values
        """
        
        confusion = SK.confusion_matrix(prediction_df["pred"], prediction_df["y"])
        #[row, column]
        print("Confusion matrix:",confusion)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        return confusion, (TP, TN, FP, FN)

    def set_prediction_threshold(self,model:T.keras.models, X_data:pd.DataFrame, y_data:pd.DataFrame, task:int, threshold:float) -> pd.DataFrame:
        """
        model: trained model
        X_data: input data
        y_data: target data
        task: default 0 (first task) but can be changed to any other task
        threshold: default 0.5 but can be changed to any other value
        """
        #print(y_data)
        prediction_train = model.predict(X_data)
        prediction_train = np.where(prediction_train > threshold, 1.0,0.0)
        predictions_df = pd.DataFrame(data={"pred":prediction_train[:,task],"y":y_data[:,task]},index=range(len(y_data)))
        predictions_df.dropna(inplace=True)
        #print(predictions_df)
        return predictions_df

    def calc_Statistics(self,TP:int,TN:int,FP:int,FN:int,prediction_df:pd.DataFrame=None, pred_y=None, obs_y=None) -> Tuple[str,tuple[float,float,float,float,float,float,float,float,float,float,float,float,float],pd.DataFrame]:
        """
        TP: True Positive
        TN: True Negative
        FP: False Positive
        FN: False Negative
        """
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * (precision * recall) / (precision + recall)
        if prediction_df is not None:
            MCC = SK.matthews_corrcoef(prediction_df["y"], prediction_df["pred"])
            kappa = SK.cohen_kappa_score(prediction_df["y"], prediction_df["pred"])
        elif pred_y is not None and obs_y is not None:
            MCC = SK.matthews_corrcoef(obs_y, pred_y)
            kappa = SK.cohen_kappa_score(obs_y, pred_y)
        SE = (TP/(TP+FN))
        SP = (TN/(TN+FP))
        PPV = (TP/(TP+FP))
        NPV = (TN/(TN+FN))
        TPR = (TP/(TP+FN))
        FPR = (FP/(FP+TN))
        #print all this stats in a table
        statistics = tabulate.tabulate([["Accuracy",accuracy],["Precision",precision],["Recall",recall],["F1",f1],["MCC",MCC],["Kappa",kappa],["SE",SE],["SP",SP],["PPV",PPV],["NPV",NPV],["TPR",TPR],["FPR",FPR]],headers=["Statistic","Value"])
        df_statistics = pd.DataFrame(data={"Statistic":["Accuracy","Precision","Recall","F1","MCC","Kappa","SE","SP","PPV","NPV","TPR","FPR"],"Value":[accuracy,precision,recall,f1,MCC,kappa,SE,SP,PPV,NPV,TPR,FPR]})
        return statistics,(accuracy, precision, recall, f1, MCC, kappa, SE, SP, PPV, NPV, TPR, FPR),df_statistics

    def calc_RegressionStatistics(self,prediction_df:pd.DataFrame=None, pred_y=None, obs_y=None)-> Tuple[str,tuple[float,float,float],pd.DataFrame]:
        """
        prediction_df: dataframe with predictions and target values
        """
        if prediction_df is not None:
            MSE = SK.mean_squared_error(prediction_df["y"], prediction_df["pred"])
            MAE = SK.mean_absolute_error(prediction_df["y"], prediction_df["pred"])
            R2 = SK.r2_score(prediction_df["y"], prediction_df["pred"])
            #print all this stats in a table
        if pred_y is not None and obs_y is not None:
            MSE = SK.mean_squared_error(obs_y, pred_y)
            MAE = SK.mean_absolute_error(obs_y, pred_y)
            R2 = SK.r2_score(obs_y, pred_y)
        statistics = tabulate.tabulate([["MSE",MSE],["MAE",MAE],["R2",R2]],headers=["Statistic","Value"])
        df_statistics = pd.DataFrame(data={"MSE":[MSE],"MAE":[MAE],"R2":[R2]})
        return statistics,(MSE, MAE, R2),df_statistics

    def set_3SigmaStats(self,obs_y,pred_y) -> Tuple[np.array,np.array]:
        data_pred = pd.DataFrame(data={"y":obs_y,"pred":pred_y}) 
        data_pred = data_pred.assign(Folds_error = [abs(data_pred['pred'][i] - data_pred['y'][i]) for i in data_pred.index])
        error = data_pred['Folds_error'].values
        error_per_line = [np.std([err,np.std(error)])*3 for err in error]

        print("Mean error: ",np.mean(error),end="\n\n")
        print("Average std error: ",np.std(error),end="\n\n")
        data_pred['3*sigma'] = error_per_line
        #print(error_per_line, end="\n\n")
        #data_pred['3*sigma'] = 
        #keep only the ones that are within +3 to -3 standard deviations.
        #Dum dum way to get rid of outliers
        drop_list = []
        for i in data_pred.index:
            if abs(data_pred['Folds_error'][i]) <= abs(data_pred['3*sigma'][i]):
                drop_list.append(i)
        print("Drop list size: ",len(drop_list),end="\n\n")
        data_pred.drop(drop_list,axis=0,errors='ignore',inplace=True) 
        pred_y = data_pred['pred'].values
        obs_y = data_pred['y'].values
        return obs_y, pred_y

    def plot_Regression(self,*X_train)-> None:
        """
        prediction_df: dataframe with predictions and target values
        """
        #concat_df = pd.concat([X_train,X_val,X_test])
        colors = ['blue','green','red','yellow','black','orange','purple','brown','pink','gray']
        labels = ['train','validation','test','train','validation','test','train','validation','test','train']
        i = 0
        for X_data in X_train:
            plt.scatter(X_data["pred"],X_data["y"],marker="o", color=colors[i], label=labels[i])
            i += 1
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()

        # Show the plot
        plt.show()

    def plot_Classification(self,cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues)-> None:
        """
        Plot a confusion matrix using Matplotlib.
        :param cm: Confusion matrix
        :param classes: List of class labels
        :param title: Title for the plot
        :param cmap: Colormap to use
        """
        fig, ax = plt.subplots()
        print(cm)
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=classes,
            yticklabels=classes,
            title=title,
            ylabel='True label',
            xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        plt.show()

    def get_modelStats(self,model:T.keras.models,X_data,y_data,tasks:int,threshold:float=0.5) -> None:
        
        if self.model_type == self.Classification:
            print("Metric for a Classification Model")
        elif self.model_type == self.Regression:
            print("Metric for a Regression Model")
        for task in range(tasks):
            try:
                if self.model_type == self.Classification:
                    classification_prediction = self.set_prediction_threshold(model, X_data, y_data, task, threshold)
                    confusion_matrix, (TP, TN, FP, FN) = self.calc_confusion_matrix(classification_prediction)
                    full_text,_ = self.calc_Statistics(TP, TN, FP, FN, classification_prediction)
                    print(full_text)
                    self.plot_Classification(confusion_matrix,["TP","FP"],f"Confusion Matrix for Task {task}")

                elif self.model_type == self.Regression:
                    regression_prediction = pd.DataFrame(data={"pred":model.predict(X_data)[:,task], "y": y_data[:,task]},index=range(len(y_data)))
                    full_text,_ = self.calc_RegressionStatistics(regression_prediction)
                    print(full_text,end="\n\n")
                    self.plot_Regression(regression_prediction)
            except:
                #X_data = Commons.ndarrayTo1Darray(Commons,X_data)
                y_data = Commons.ndarrayTo1Darray(Commons,y_data)
                
                if self.model_type == self.Classification:
                    classification_prediction = self.set_prediction_threshold(model, X_data, y_data, task, threshold)
                    confusion_matrix, (TP, TN, FP, FN) = self.calc_confusion_matrix(classification_prediction)
                    full_text,_ = self.calc_Statistics(TP, TN, FP, FN, classification_prediction)
                    print(full_text)
                    self.plot_Classification(confusion_matrix,["TP","FP"],f"Confusion Matrix for Task {task}")
                
                elif self.model_type == self.Regression:
                    regression_prediction = pd.DataFrame(data={"pred":model.predict(X_data), "y": y_data},index=range(len(y_data)))
                    full_text,_ = self.calc_RegressionStatistics(regression_prediction)
                    print(full_text,end="\n\n")
                    self.plot_Regression(regression_prediction)
                        
class TS_Helper(Statistics):
    
    def __init__(self):
        """Define model type with Ts_Helper.model_type = Ts_Helper.classification or Ts_Helper.regression"""
        super().__init__()
    #super = super(Statistics,Statistics())

    MISSING_LABEL_FLAG = -1
    BinaryCrossentropy = T.keras.losses.BinaryCrossentropy()
    CategoricalCrossentropy = T.keras.losses.CategoricalCrossentropy()
    MeanSquaredError = T.keras.losses.MeanSquaredError()
    SparseCategoricalCrossentropy = T.keras.losses.SparseCategoricalCrossentropy() 

    def classification_loss(self,loss_function:Callable,mask_value:int = MISSING_LABEL_FLAG)-> Callable: #used in multitask classification models (customizable)
    
        """Builds a loss function that masks based on targets
        Args:
            loss_function: The loss function to mask
            mask_value: The value to mask in the targets
        Returns:
            function: a loss function that acts like loss_function with masked inputs
        """

        def masked_loss_function(y_true:float, y_pred:float):  
            y_true = float(y_true)
            y_pred = float(y_pred)
            y_true = T.where(T.math.is_nan(y_true), T.zeros_like(y_true), y_true)
            dtype = K.floatx()
            mask = T.cast(T.not_equal(y_true, mask_value), dtype)
            return loss_function(y_true * mask, y_pred * mask)

        return masked_loss_function

    def regression_loss(self,y_true:float, y_pred:float) -> float:  #Used in multitask regression models (MSE metric)
        y_true = T.where(T.math.is_nan(y_true), T.zeros_like(y_true), y_true)
        loss = T.reduce_mean(
        T.where(T.equal(y_true, 0.0), y_true,
        T.square(T.abs(y_true - y_pred))))
        
        return loss     
    # adaptative learning rate using during training
    def get_lr_metric(self,optimizer:T.optimizers.Optimizer):
        def lr(y_true, y_pred):
            return optimizer.lr
            
        return lr
    
    def set_gpu_fraction(self, gpu_fraction:int,sess=None)->T.compat.v1.Session:
        """Set the GPU memory fraction for the application.

        Parameters
        ----------
        sess : a session instance of TensorFlow
            TensorFlow session
        gpu_fraction : a float
            Fraction of GPU memory, (0 ~ 1]

        References
        ----------
        - `TensorFlow using GPU <https://www.tensorflow.org/versions/r0.9/how_tos/using_gpu/index.html>`_
        """
        print("  tensorlayer: GPU MEM Fraction %f" % gpu_fraction)
        
        gpu_options = T.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        sess = T.compat.v1.Session(config = T.compat.v1.ConfigProto(gpu_options = gpu_options))
        print("Num GPUs Available: ", len(T.config.list_physical_devices('GPU')))
        return sess 
    
    def plot_history(self,history:dict) -> None:
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch

        plt.figure(figsize=(13, 9))
        plt.subplot(221)
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.plot(hist['epoch'], hist['val_loss'], label = 'Val Error')
        plt.plot(hist['epoch'], hist['loss'], label='Train Error')
        plt.legend()

        plt.subplot(222)
        plt.xlabel('Epoch')
        plt.ylabel('lr')
        plt.plot(hist['epoch'], hist['lr'], label='lr')
        plt.legend()

        plt.show()

    def get_modelStatsFor_nSplits(self,model:T.keras.models,tasks:int=1,threshold:float=0.5,**XandY_data) -> None:
        """
        Assumes that the keys of XandY_data are passed in order of X_data_1,y_data_1, . . . X_data_n,y_data_n.

        Where X_data_1 is the X_data for the first split and y_data_1 is the y_data for the first split, and so on.

        We assume that the number of X_data and y_data are equal.

        obs: X_data could be X_train and y_data could be y_train
        """
        X_data = []
        y_data = []
        predictions = []
        for k,v in XandY_data.items():
            
            if k.startswith("X"):
                X_data.append(v)
            
            elif k.startswith("y"):
                y_data.append(v)
        
        if len(X_data) == len(y_data):
            
            for i in range(len(X_data)):
                self.get_modelStats(model,X_data[i],y_data[i],tasks,threshold)
                
                if self.model_type == self.Regression:
                    y_data[i] = Commons.ndarrayTo1Darray(Commons,y_data[i])
                    
                    for task in range(tasks):
                        prediction = pd.DataFrame(data={"pred":model.predict(X_data[i])[:,task],"y":y_data[i]},index=range(len(y_data[i])))
                        predictions.append(prediction)
            
            if self.model_type == self.Regression:
                    self.plot_Regression(*predictions)
            #self.plot_Regression(predictions)
        else:
            raise Exception("X and y data must be equal in length")

class ML_Helper(Statistics):
    def __init__(self):
        super().__init__()
    
    
    def get_ML_StatsForNSplits(self,model:T.keras.models,**XandY_data) -> Tuple[str,pd.DataFrame]:
        X_data = []
        y_data = []
        texts = ""
        for k,v in XandY_data.items():
            if k.startswith("X"):
                X_data.append(v)
            elif k.startswith("y"):
                y_data.append(v)
        
        if len(X_data) == len(y_data):
            for i in range(len(X_data)):
                
                if self.model_type == self.Regression:
                    y_pred = model.predict(X_data[i])
                    text,(_,_,_),df = self.calc_RegressionStatistics(obs_y=y_data[i],pred_y=y_pred)

                    print("Before 3 Sigma:\n",text,end="\n\n")
                    texts += "Before 3 Sigma:\n"+text+"\n\n"
                    print("After 3 Sigma:\n")
                    
                    y_data[i], y_pred = self.set_3SigmaStats(obs_y=y_data[i],pred_y=y_pred)
                    text,(_,_,_),df_3S = self.calc_RegressionStatistics(obs_y=y_data[i],pred_y=y_pred)
                    
                    print(text,end="\n\n")
                    texts += "After 3 Sigma:\n"+text+"\n\n"
                    df = pd.concat([df,df_3S],axis=1)
                elif self.model_type == self.Classification:
                    
                    y_pred = (model.predict_proba(X_data[i])[:,1] >= 0.5).astype(bool)
                    prediction_df = pd.DataFrame({'y': y_data[i], 'pred': y_pred})
                    confusion, (TP, TN, FP, FN) = self.calc_confusion_matrix(prediction_df)
                    text,_,df =  self.calc_Statistics(TP, TN, FP, FN, prediction_df)
                    
                    print(text)
                    texts += text

                    self.plot_Classification(confusion,[['T','F'],["P","N"]],title='Test set')

        else:
            raise Exception("X and y data must be equal in length")
        return texts,df
    
class Models():

    def __init__(self):
        pass
    KEY_WORDS = GLOBALS
    AVAILABLE_MODELS = KEY_WORDS.AVAILABLE_MODELS
    CLASS = KEY_WORDS.CLASSIFIER
    PARAMS = KEY_WORDS.PARAMS
    LGBM = AVAILABLE_MODELS[0]
    RF = AVAILABLE_MODELS[1]
    SVM = AVAILABLE_MODELS[2]
    KMEANS = AVAILABLE_MODELS[3]
    GAUSSIAN_MIX = AVAILABLE_MODELS[4]
    DB_SCAN = AVAILABLE_MODELS[5]
    OPTICS = AVAILABLE_MODELS[6]
    AGG_CLUSTER = AVAILABLE_MODELS[7]
    AFFINITY_PROP = AVAILABLE_MODELS[8]
    BIRCH = AVAILABLE_MODELS[9]
    
    def get_Model(self,model_name:str):
        if model_name is None:
            raise Exception("Please provide a model name")
        models = [(i,model.lower()) for i,model in enumerate(self.AVAILABLE_MODELS)]
        for i,model in models:
            if model == model_name.lower():
                return self.AVAILABLE_MODELS[i]
            if i == len(models)-1:
                raise Exception("Model not found")
    
class Model_Generator(Models):
    """
    Class with dictionaries to input in BayesSearchcv
    Helper methods to get desired model with minimum difficulty

    Each model contains neccessary params to input in a Bayesean Search
    """
    def __init__(self):
        super().__init__()
    
    def set_Model(self, model_name: str) -> dict:
        Models: dict = {

        #Machine Learning Models
        self.LGBM:{
            self.CLASS:lgb.LGBMClassifier(),                   
            self.PARAMS:{
                'learning_rate': (0.01, 0.1, 'uniform'), 
                'num_leaves': (1, 15),
                'n_estimators': (2, 50), 
                'max_depth': (1, 10),
                'subsample': (0.1, 0.3), 
            },
        },
        self.RF:{
            self.CLASS:RandomForestClassifier(),
            self.PARAMS:{
            'max_features': ['auto', 'sqrt'],
            'n_estimators': [2, 150],
            "max_depth": [2, 10],
            },
        },
        self.SVM:{
            self.CLASS:SVC(),
            self.PARAMS:{
                'C': (1e-7, 1e-1, 'uniform'),
                #'gamma': (0.01, 1.0, 'log-uniform'),
                'degree': (1, 8),
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            },
        },

        #CLUSTERING ALGORITHMS
        self.KMEANS:{
            self.CLASS:KMeans(),
            self.PARAMS:{
                "n_clusters":[1,100],
                "tol":[1e-6,1e-2],
                "algorithm":["lloyd", "elkan", "auto", "full"]
            }        
        },
        self.DB_SCAN:{
            self.CLASS:DBSCAN(),
            self.PARAMS:{
                "eps":[0.1,1.0],
                "min_samples":[1,10],
                "metric":["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan"],
                "algorithm":["auto", "ball_tree", "kd_tree", "brute"],
                "leaf_size":[5,100],
            }
        },
        self.AGG_CLUSTER:{
            self.CLASS:AgglomerativeClustering(),
            self.PARAMS:{
                "n_clusters":[1,100],
                "linkage":["ward", "complete", "average", "single"],
                "metric":["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan"],
            }
        },
        self.AFFINITY_PROP:{
            self.CLASS:AffinityPropagation(),
            self.PARAMS:{
                "damping":[0.5,1.0],
                "max_iter":[100,1000],
                "convergence":[15],
                "affinity":["euclidean", "precomputed"],
            }
        },
        self.BIRCH:{
            self.CLASS:Birch(),
            self.PARAMS:{
                "threshold":[0.1,1.0],
                "branching_factor":[10,200],
                "n_clusters":[1,100],
            }
        },
        self.GAUSSIAN_MIX:{
            self.CLASS:GaussianMixture(),
            self.PARAMS:{
                "n_components":[1,10],
                "covariance_type":["full", "tied", "diag", "spherical"],
                "tol":[1e-6,1e-1],
                "reg_covar":[1e-6,1e-1],
                "max_iter":[100,1000],
                "n_init":[1,10],
                "init_params":["kmeans", "random"],
            }
        },
        self.OPTICS:{
            self.CLASS:OPTICS(),
            self.PARAMS:{
                "min_samples":[1,10],
                "max_eps":[0.1,1.0],
                "metric":["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan"],
                "algorithm":["auto", "ball_tree", "kd_tree", "brute"],
                "leaf_size":[5,100],
                "cluster_method":["xi", "dbscan"]
            }
        },
    }
        return Models[model_name]
    
    def get_AvailableModels(self) -> List[str]:
        print("Available Models: " + str(self.AVAILABLE_MODELS))
        return self.AVAILABLE_MODELS
    
    def get_Model(self, model_name: str):
        return super().get_Model(model_name)

    def get_buildParams(self,model_name:str) -> Tuple[object,dict]:
        """
        Returns the model and its params respectively, if available 
        """
        model = self.get_Model(model_name)
        return self.set_Model(model)[self.CLASS],self.set_Model(model)[self.PARAMS]     

class BayesSearch_Helper(Model_Generator,ML_Helper):
    """
    Class with helper methods to run BayesSearchCV on models contained in Model_Generator
    """

    def  __init__(self):
        super().__init__()
    
    def get_BestParams(self,model_name:str,X:np.ndarray,y:np.ndarray) -> Tuple[dict,float]:
        """
        Builds a BayesSearchCV object with the model and its params
        """
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
        scorer = make_scorer(mean_squared_error)
        model,params = self.get_buildParams(model_name)
        clf = BayesSearchCV(
            estimator=model,
            search_spaces=params,
            cv=cv,
            n_jobs=-1,
            n_iter=2,
            verbose=0,
            refit=True,
            scoring = scorer,
            random_state=42
        )
        clf.fit(X,y)
        return clf.best_params_,clf.best_score_
    
    def build(self,model_name,params,x_data,y_data):
        """
        Builds a model with the best parameters found by BayesSearchCV
        """
        model,_ = self.get_buildParams(model_name)
        model = model.set_params(**params)
        model.fit(x_data,y_data)
        return model
    
    def get_Stats(self,model_name:str,X:np.ndarray,y:np.ndarray) -> Tuple[BayesSearchCV,dict]:
        """
        Returns the best parameters and the best score of the model
        """
        best,_ = self.get_BestParams(model_name,X,y)
        model = self.build(model_name,best,X,y)
        text,df = self.get_ML_StatsForNSplits(model,X_data=X,y_data=y)
        return text,df


