import pandas as pd
import pickle
import random
recipes = pd.read_csv("Recipes.csv")
defects = pd.read_csv("Defects.csv")

df = pd.pivot_table(recipes, index="HEAT_ID", columns=['Material_Code'], values='WEIGHT')
df.reset_index(inplace=True)
df = pd.merge(defects, df, on='HEAT_ID')
df = df.fillna(0)
df = df.drop(columns=['PIECE_ID', 'DEFECT_NAME', 'DEFECT_GROUP_Name'])
df = df.drop_duplicates()
df['Sum'] = df.iloc[:, 2:30].sum(axis=1)
dfct = df['DEFECT_TYPE']
df = df.drop(columns=['DEFECT_TYPE'])
df['DEFECT_TYPE'] = dfct
dfct = df['DEFECT_GROUP_ID']
df = df.drop(columns=['DEFECT_GROUP_ID'])
df['DEFECT_GROUP_ID'] = dfct
recp = recipes.Material_Code.unique()
temp_data = {k: g["DEFECT_TYPE"].tolist() for k,g in df.groupby(["HEAT_ID","DEFECT_GROUP_ID"])}

def preprocess(recipes, defects):
    df = pd.pivot_table(recipes, index="HEAT_ID", columns=['Material_Code'], values='WEIGHT')
    df.reset_index(inplace=True)
    df = pd.merge(defects, df, on = 'HEAT_ID')
    df = df.fillna(0)
    df = df.drop(columns=['PIECE_ID', 'DEFECT_NAME', 'DEFECT_GROUP_Name'])
    df = df.drop_duplicates()
    df['Sum'] = df.iloc[:, 2:30].sum(axis=1)
    df = df.sample(frac=1)
    return df

def organize_data(df):
    dfct = df['DEFECT_TYPE']
    df = df.drop(columns=['DEFECT_TYPE'])
    df['DEFECT_TYPE'] = dfct
    dfct = df['DEFECT_GROUP_ID']
    df = df.drop(columns=['DEFECT_GROUP_ID'])
    df['DEFECT_GROUP_ID'] = dfct
    return df

def testing_data(heat_Id, mat_list, recp):
    mat_list = mat_list.split(',')
    x = []
    x_pred = []
    mat_code = []
    mat_weight = []
    # split mat code and weight from mat_list
    for i in range(len(mat_list)):
        mat_code.append(mat_list[i].split('--')[0].strip())
        mat_weight.append(float(mat_list[i].split('--')[1].strip()))
    mat_code = [int(i) for i in mat_code]
    recp = sorted(recp)
    # if user inputed mat code is in relation, mat weight will be save, if not, save as 0.0
    c = 0
    for i in recp:
        if i in mat_code:
            x.append(mat_weight[c])
            c+=1
        else:
            x.append(0.0)
    x.append(sum(x))
    x.insert(0, heat_Id)
    x_pred.append(x)
    return x_pred


# load the model from disk
def get_result(result, heatId, temp_data):
    heatId = int(heatId)
    final_result = temp_data[(heatId, result)]
    final_result = random.choice(final_result)
    return final_result