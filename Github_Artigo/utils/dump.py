def unused_code(self):
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