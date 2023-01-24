import pandas as pd
import numpy as np
import tensorflow as T
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
import itertools

from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import SVG, Image
IPythonConsole.molSize = (400,400)
import matplotlib.pyplot as plt
import tabulate
import sklearn.metrics as SK
from tensorflow.keras import backend as K
from tensorflow.keras.losses import *
import seaborn as sns
from typing import List, Set, Tuple, Union,Callable, Optional, Dict, Any
from dataclasses import dataclass
import PIL
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
class Commons():
    def __init__(self):
        pass
    def load_dataset(self, dataset:pd.DataFrame,smiles_col:str ,task_start:int=0, number_of_tasks:int=1) -> tuple:
        task_end = task_start + number_of_tasks
        
        df = pd.read_csv(dataset, sep=",", low_memory=False)
        y_train = np.array(df.iloc[:,task_start:task_end].values)
        smiles = df[smiles_col].values
        print(f"Loaded dataset {dataset} with shape: {df.shape}")
        return df,y_train,smiles
    
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

class TS_Helper():
    
    def __init__(self):
        """Define model type with Ts_Helper.model_type = Ts_Helper.classification or Ts_Helper.regression"""
        pass
    
    Classification = GLOBALS.CLASSIFICATION
    Regression = GLOBALS.REGRESSION

    model_type: int = Classification or Regression
    assert model_type in [Classification, Regression], "model_type must be either Ts_Helper.Classification or Ts_Helper.Regression"
    
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

    def set_prediction_threshold(self,model:T.keras.models.Sequential(),X_data:pd.DataFrame,y_data:pd.DataFrame,task:int,threshold:float) -> pd.DataFrame:
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

    def calc_Statistics(self,TP:int,TN:int,FP:int,FN:int,prediction_df) -> Tuple[str,tuple[float,float,float,float,float,float,float,float,float,float,float,float,float]]:
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
        MCC = SK.matthews_corrcoef(prediction_df["y"], prediction_df["pred"])
        kappa = SK.cohen_kappa_score(prediction_df["y"], prediction_df["pred"])
        SE = (TP/(TP+FN))
        SP = (TN/(TN+FP))
        PPV = (TP/(TP+FP))
        NPV = (TN/(TN+FN))
        TPR = (TP/(TP+FN))
        FPR = (FP/(FP+TN))
        #print all this stats in a table
        statistics = tabulate.tabulate([["Accuracy",accuracy],["Precision",precision],["Recall",recall],["F1",f1],["MCC",MCC],["Kappa",kappa],["SE",SE],["SP",SP],["PPV",PPV],["NPV",NPV],["TPR",TPR],["FPR",FPR]],headers=["Statistic","Value"])
        return statistics,(accuracy, precision, recall, f1, MCC, kappa, SE, SP, PPV, NPV, TPR, FPR)

    def calc_RegressionStatistics(self,prediction_df)-> Tuple[str,tuple[float,float,float]]:
        """
        prediction_df: dataframe with predictions and target values
        """
        MSE = SK.mean_squared_error(prediction_df["y"], prediction_df["pred"])
        MAE = SK.mean_absolute_error(prediction_df["y"], prediction_df["pred"])
        R2 = SK.r2_score(prediction_df["y"], prediction_df["pred"])
        #print all this stats in a table
        statistics = tabulate.tabulate([["MSE",MSE],["MAE",MAE],["R2",R2]],headers=["Statistic","Value"])
        return statistics,(MSE, MAE, R2)

    def plot_Regression(self,X_train,X_val,X_test)-> None:
        """
        prediction_df: dataframe with predictions and target values
        """
        #concat_df = pd.concat([X_train,X_val,X_test])

        plt.scatter(X_train["pred"], X_train["y"],marker="o", color='blue', label='train')
        #plt.plot(X_train["pred"], X_train["y"], color='blue')

        # Plot the validation data
        plt.scatter(X_val["pred"],X_val["y"],marker="o", color='green', label='validation')
        #plt.plot(X_val["pred"],X_val["y"], color='green')

        # Plot the test data
        plt.scatter(X_test["pred"],X_test["y"],marker="o", color='red', label='test')
        
        #plt.plot(np.mean(concat_df["pred"]), color='purple', label='regression line')

        # Add labels and legend to the plot
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

    def get_RegressionStats(self,model:T.keras.models.Sequential(),X_train,y_train,X_test,y_test,X_val,y_val,task:int) -> Tuple[str,tuple[float,float,float]]:
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
    
    def get_ClassificationStats(self,model,X_train:pd.DataFrame,y_train,X_test:pd.DataFrame,y_test,X_val:pd.DataFrame,y_val,task:int,threshold:float) -> Tuple[str, tuple[float,float,float],Tuple[np.ndarray],Tuple[np.ndarray],Tuple[np.ndarray]]:
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
    
    def get_modelStats(self,model:T.keras.models.Sequential(),X_train,y_train,X_test,y_test,X_val,y_val,tasks:int,threshold:float=0.5) -> None:
        if self.model_type == 0:
            print("Metric for a Classification Model")
        else:
            print("Metric for a Regression Model")
        for task in range(tasks):
            
            if self.model_type == 0:
                full_text,cross_val,train_confusion_matrix,val_confusion_matrix,test_confusion_matrix,   = self.get_ClassificationStats(model,X_train,y_train,X_test,y_test,X_val,y_val,task,threshold)
                print(full_text)
                self.plot_Classification(train_confusion_matrix,["TP","FP"],f"Confusion Matrix for Training in Task {task}")
                self.plot_Classification(val_confusion_matrix,["TP","FP"],f"Confusion Matrix for Validation in Task {task}")
                self.plot_Classification(test_confusion_matrix,["TP","FP"],f"Confusion Matrix for Testing in Task {task}")

            if self.model_type == 1:
                full_text,cross_val = self.get_RegressionStats(model,X_train,y_train,X_test,y_test,X_val,y_val,task)
                print(full_text)
                self.plot_Regression(cross_val[0],cross_val[1],cross_val[2])              

class Model_Generator(TS_Helper):
    def __init__(self,):
        super().__init__()
    
        Models = {
        "Dense":T.keras.models.Sequential([
            T.keras.layers.Dense(64,activation="relu"),
            T.keras.layers.Dense(64,activation="relu"),
            T.keras.layers.Dense(1,activation="sigmoid")
        ]),
        "LSTM":T.keras.models.Sequential([
            T.keras.layers.LSTM(64,activation="relu",return_sequences=True),
            T.keras.layers.LSTM(64,activation="relu"),
            T.keras.layers.Dense(1,activation="sigmoid")
        ]),
        "GRU":T.keras.models.Sequential([
            T.keras.layers.GRU(64,activation="relu",return_sequences=True),
            T.keras.layers.GRU(64,activation="relu"),
            T.keras.layers.Dense(1,activation="sigmoid")
        ]),
        "Conv1D":T.keras.models.Sequential([
            T.keras.layers.Conv1D(64,3,activation="relu"),
            T.keras.layers.MaxPool1D(3),
            T.keras.layers.Conv1D(64,3,activation="relu"),
            T.keras.layers.MaxPool1D(3),
            T.keras.layers.Flatten(),
            T.keras.layers.Dense(1,activation="sigmoid")
        ]),
        "Conv2D":T.keras.models.Sequential([
            T.keras.layers.Conv2D(64,3,activation="relu"),
            T.keras.layers.MaxPool2D(3),
            T.keras.layers.Conv2D(64,3,activation="relu"),
            T.keras.layers.MaxPool2D(3),
            T.keras.layers.Flatten(),
            T.keras.layers.Dense(1,activation="sigmoid")
        ]),
        "Conv3D":T.keras.models.Sequential([
            T.keras.layers.Conv3D(64,3,activation="relu"),
            T.keras.layers.MaxPool3D(3),
            T.keras.layers.Conv3D(64,3,activation="relu"),
            T.keras.layers.MaxPool3D(3),
            T.keras.layers.Flatten(),
            T.keras.layers.Dense(1,activation="sigmoid")
        ]),
        "RNN":T.keras.models.Sequential([
            T.keras.layers.SimpleRNN(64,activation="relu",return_sequences=True),
            T.keras.layers.SimpleRNN(64,activation="relu"),
            T.keras.layers.Dense(1,activation="sigmoid")
        ]),
        "Bidirectional":T.keras.models.Sequential([
            T.keras.layers.Bidirectional(T.keras.layers.LSTM(64,activation="relu",return_sequences=True)),
            T.keras.layers.Bidirectional(T.keras.layers.LSTM(64,activation="relu")),
            T.keras.layers.Dense(1,activation="sigmoid")
        ]),
        "TimeDistributed":T.keras.models.Sequential([
            T.keras.layers.TimeDistributed(T.keras.layers.Dense(64,activation="relu")),
            T.keras.layers.TimeDistributed(T.keras.layers.Dense(64,activation="relu")),
            T.keras.layers.Dense(1,activation="sigmoid")
        ]),
        "Attention":T.keras.models.Sequential([
            T.keras.layers.Attention(),
            T.keras.layers.Dense(1,activation="sigmoid")
        ]),
        "Transformer":T.keras.models.Sequential([
            T.keras.layers.Transformer(),
            T.keras.layers.Dense(1,activation="sigmoid")
        ]),
        "TransformerEncoder":T.keras.models.Sequential([
            T.keras.layers.TransformerEncoder(),
            T.keras.layers.Dense(1,activation="sigmoid")
        ]),
        "TransformerDecoder":T.keras.models.Sequential([
            T.keras.layers.TransformerDecoder(),
            T.keras.layers.Dense(1,activation="sigmoid")
        ]),
        "TransformerEncoderLayer":T.keras.models.Sequential([
            T.keras.layers.TransformerEncoderLayer(),
            T.keras.layers.Dense(1,activation="sigmoid")
        ]),
        "TransformerDecoderLayer":T.keras.models.Sequential([
            T.keras.layers.TransformerDecoderLayer(),
            T.keras.layers.Dense(1,activation="sigmoid")
        ]),
        "TransformerEncoderBlock":T.keras.models.Sequential([
            T.keras.layers.TransformerEncoderBlock(),
            T.keras.layers.Dense(1,activation="sigmoid")
        ]),
        "TransformerDecoderBlock":T.keras.models.Sequential([
            T.keras.layers.TransformerDecoderBlock(),
            T.keras.layers.Dense(1,activation="sigmoid")
        ]),
        "TransformerEncoderStack":T.keras.models.Sequential([
            T.keras.layers.TransformerEncoderStack(),
            T.keras.layers.Dense(1,activation="sigmoid")
        ]),
        "TransformerDecoderStack":T.keras.models.Sequential([
            T.keras.layers.TransformerDecoderStack(),
            T.keras.layers.Dense(1,activation="sigmoid")
        ]),
        "TransformerEncoderHead":T.keras.models.Sequential([
            T.keras.layers.TransformerEncoderHead(),
            T.keras.layers.Dense(1,activation="sigmoid")
        ]),
        "TransformerDecoderHead":T.keras.models.Sequential([
            T.keras.layers.TransformerDecoderHead(),
            T.keras.layers.Dense(1,activation="sigmoid")
        ]),
    }

