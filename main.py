from numpy.random.mtrand import logistic
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from matplotlib.pyplot import figure
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image


st.title("Machine Learning for Everyone")

st.write("""

### Upload different datasets and see how the models perform
""")
# img = Image.open("images/download.jpg")
# st.image(img)

st.sidebar.subheader("Settings")
uploaded_file = st.sidebar.file_uploader(label="Upload File", type=['csv','xlsx'])
classsifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "Logistic", "SVC", "Random Forest"))

def download_file(df, types, new_types, extension):
  for i, col in enumerate(df.columns):
    new_type = types[new_types[i]]
    if new_type:
      try:
        df[col] = df[col].astype(new_type)
      except:
        st.write('Could not convert', col, 'to', new_types[i])

def transform(df):
  # Select sample size
  st.write('\n')
  st.write('\n')
  st.write('## Data')
  frac = st.slider('Random sample the data (%)', 1, 100, 100)
  if frac < 100:
    df = df.sample(frac=frac/100)
     # Select columns
  cols = st.multiselect('Choose Columns', 
                        df.columns.tolist(),
                        df.columns.tolist())
  df = df[cols]
  types = {'-':None
           ,'Boolean': '?'
           ,'Byte': 'b'
           ,'Integer':'i'
           ,'Floating point': 'f' 
           ,'Date Time': 'M'
           ,'Time': 'm'
           ,'Unicode String':'U'
           ,'Object': 'O'}
  new_types = {}
  expander_types = st.sidebar.expander('Convert Data Types')
  for i, col in enumerate(df.columns):
    txt = 'Convert {} from {} to:'.format(col, df[col].dtypes)
    expander_types.markdown(txt, unsafe_allow_html=True)
    new_types[i] = expander_types.selectbox('Field to be converted:'
                                            ,[*types]
                                            ,index=0
                                            ,key=i)
  st.text(" \n") #break line

  download_file(df, types, new_types, "csv")


  return df
# def explore(df):
#   pr = ProfileReport(df, explorative=True)
#   st_profile_report(pr)

def explore(df):
    sns.heatmap(df.corr())
    plt.show()

def train_test_split(df):
    st.write('## Label Column')
    label_name = st.selectbox('Choose the label column',df.columns.tolist()[::-1])
    list_features = []
    for i in df.columns:
      
        if len(label_name) > 0:
            if i != label_name:
       
                list_features.append(i)



    return label_name,list_features


def get_params(name):
    params = {}
    if name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    elif name == 'SVC' or name == 'Logistic':
        C = st.sidebar.slider('C', 0.1, 10.0)
        params['C'] = C
    elif name == 'Random Forest':
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params
def get_classifier(classifier_name, params):
    if classsifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif classifier_name == 'SVC':
        clf = SVC(C=params["C"])
    elif classifier_name == 'Random Forest':
        clf = RandomForestClassifier(n_estimators=params["n_estimators"])
    elif classsifier_name == 'Logistic':
        clf = LogisticRegression()

    return clf
global data 


if uploaded_file is not None:
    
    try:
        data = pd.read_csv(uploaded_file)

    except Exception as e:
        st.write('## (Error in data)')
        

        
    except ValueError as e:
        data = pd.read_excel(uploaded_file)


try:
    params = get_params(classsifier_name)
    
    data = transform(data)
    st.write(data)

    
    
    

    # ax.set_facecolor("black")
    # ax.plot(data['Iris-setosa'], data['Iris-setosa'])
    # ax.plot(data['Iris-setosa'], data['Iris-setosa'])
    # ax.legend()
    

    #st.bar_chart(label,width=50, height=50)
    st.write('\n')
    st.write('\n')

    st.write('## Plots')
    col = st.multiselect('Line chart', 
                        data.columns.tolist(),
                        data.columns.tolist())
    columns = []
    for i in col:
        if data[i].dtype == 'float64' or data[i].dtype == 'int64':
            columns.append(i)


    st.line_chart(data[columns])
    col = st.multiselect('Bar chart', 
                        data.columns.tolist(),
                        data.columns.tolist())
    columns = []
    for i in col:
        if data[i].dtype == 'float64' or data[i].dtype == 'int64':
            columns.append(i)


    st.area_chart(data[columns])
    
    print(columns)
    #explore(data)
    label,features = train_test_split(data)
    label = data[label]
    data = data[features]
    if len(features) > 0:
        nonObjects = []
        objects = []
        #print('feat')
        #print(features)
        for i in features:
            
            if data[i].dtype == 'float64' or data[i].dtype == 'int64':
                nonObjects.append(i)
            else:
                if len(data[i].value_counts()) <=10:
                #print(len(data[i].value_counts()))
                    objects.append(i)
        total = objects
        print(total)
        st.write("## Feature Columns")
        # print('***')
        # print(total)
        # print(label)
        st.write('If there are categorical columns then you can use one hot encoding')
        st.write('### One Hot Encoding')
        st.write('Only categorical columns can be selected')
        col = st.multiselect('', 
                        total,
                        total)
        one_hot_encoded_data = pd.get_dummies(data, columns = col)

        
        st.write(one_hot_encoded_data)
        st.write('\n')
        st.write('\n')

        col = st.multiselect('Bar chart', 
                        data.columns.tolist(),
                        data.columns.tolist())
        columns = []
        for i in col:
            if data[i].dtype == 'float64' or data[i].dtype == 'int64':
                columns.append(i)
        st.area_chart(one_hot_encoded_data[columns])
        #st.write(data[nonObjects])

        clf = get_classifier(classsifier_name,params)
        st.write(one_hot_encoded_data.shape)
        
        
        data = data.sample(frac=1)
        X = one_hot_encoded_data
        y = label
        train_pct_index = int(0.8 * len(X))
        X_train, X_test = X[:train_pct_index], X[train_pct_index:]
        y_train, y_test = y[:train_pct_index], y[train_pct_index:]
        sc_x = StandardScaler()
        X_train = sc_x.fit_transform(X_train) 
        X_test = sc_x.transform(X_test)
        print(y_train)
        st.write('### Standardized Columns')
        st.write(X_train)
  
        #X_train, X_test, y_train, y_test = train_test_split(X,y.any(),test_size=0.2)

        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        st.write('## Accuracy')
        st.write('\n')
        st.write('\n')

        st.write(accuracy_score(y_test,y_pred))
        st.write('\n')
        st.write('\n')
        st.write('### Confusion Matrix')
        st.write(confusion_matrix(y_test, y_pred))
        st.write('\n')
        st.write('\n')
        st.write('### Classifcation Report')
        '(classification report)  ',classification_report(y_test, y_pred)
      


    
    

except Exception as e:
    st.text(" \n") #break line
    st.text(" \n") #break line
    st.text(" \n") #break line
    st.text(" \n") #break line
    st.text(" \n") #break line
    st.text(" \n") #break line
    if uploaded_file != None:
        st.write("## There was in error")