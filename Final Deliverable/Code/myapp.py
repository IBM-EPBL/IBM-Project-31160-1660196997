from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, render_template
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
app = Flask(__name__)
@app.route("/")
def home():
    return render_template("Login.html")

@app.route("/index",methods=["POST","GET"])
def index():
    return render_template("index.html")


@app.route("/data",methods=["POST","GET"])
def result():
    temp=[]
    columns=['X','Y','OBJECTID','FEATURE_ID','ZIP','LATITUDE','LONGITUDE','CENSUS_CODE']
    temp.append(int(request.form['X']))
    temp.append(int(request.form['Y']))
    temp.append(int(request.form['objectid']))
    temp.append(int(request.form['featureid']))
    temp.append(int(request.form['zipcode']))
    temp.append(int(request.form['latitude']))
    temp.append(int(request.form['longitude']))
    temp.append(int(request.form['censuscode']))
    df = pd.read_csv('DHL_Facilities.csv')
    X = df.iloc[:,0:4] #Geo-Codes, ObjectID, FeatureID
    Y = df.iloc[:,9:12] #Latitude, Longitude
    Z = df.iloc[:,14] #ZipCode
    X = pd.concat([X,Y,Z],axis = 1)
    Y = df.iloc[:,7]
    X = X.replace('Not Available',0)
    X = pd.DataFrame(X)
    df1=pd.DataFrame(columns=columns)
    df2=pd.concat((df1,pd.DataFrame(data=[temp],columns=columns)))
    classifier=RandomForestClassifier(n_estimators=50, random_state=0)  
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state = 0)
    classifier.fit(x_train,y_train)
    res = classifier.predict(df2)
    return render_template("data.html",result = res)

    if __name__=='__main__':
    app.run()
