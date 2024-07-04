from flask import Flask, request, jsonify, render_template, redirect, flash, send_file, Response
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import preprocessing
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
#Thresholds
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import warnings
warnings.filterwarnings("ignore") 

mpl.rcParams["figure.figsize"] = [7, 7]
mpl.rcParams["figure.autolayout"] = True

app = Flask(__name__)

path = "creditcard.csv"
data=pd.read_csv(path)
x_dummy=data.drop(columns='Class', axis=1)
y=data['Class']
scaler=StandardScaler()
x=scaler.fit_transform(x_dummy)
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.20, random_state=123)

def logic_regression(x_train, y_train, x_test):
      lr=LogisticRegression()
      lr.fit(x_train, y_train)
      y_train_pred=lr.predict(x_train)
      y_train_cl_report=classification_report(y_train, y_train_pred, target_names = ['No Fraud', 'Fraud'], output_dict=True)
      y_test_pred=lr.predict(x_test)
      y_test_cl_report=classification_report(y_test, y_test_pred, target_names = ['No Fraud', 'Fraud'], output_dict=True)
    
      return y_test_pred, lr, y_train_cl_report, y_test_cl_report

def KNeighbors(x_train, y_train, x_test):
  Kneib=KNeighborsClassifier(n_neighbors=4)
  Kneib.fit(x_train, y_train)
  y_train_pred=Kneib.predict(x_train)
  y_train_cl_report=classification_report(y_train, y_train_pred, target_names = ['No Fraud', 'Fraud'], output_dict=True)
  y_test_pred=Kneib.predict(x_test)
  y_test_cl_report=classification_report(y_test, y_test_pred, target_names = ['No Fraud', 'Fraud'], output_dict=True)
  return y_test_pred,Kneib, y_train_cl_report, y_test_cl_report

y_test_pred, lr, log_tr_re, log_te_re = logic_regression(x_train, y_train, x_test)
log_tr_re = pd.DataFrame(log_tr_re).transpose()
log_te_re = pd.DataFrame(log_te_re).transpose()

y_test_pred, Kneib,  knn_tr_re, knn_te_re=KNeighbors(x_train, y_train, x_test)
knn_tr_re = pd.DataFrame(knn_tr_re).transpose()
knn_te_re = pd.DataFrame(knn_te_re).transpose()

def makeplot():
      i=0
      fig1 = plt.figure(figsize=(6,6))
      sns.set_theme(style="darkgrid")
      plt.scatter(y_test, y_test_pred)
      sns.regplot(x=y_test, y=y_test_pred,logistic=True)
      plt.xlabel('Actual Frauds')
      plt.ylabel('Predicted Frauds')
      plt.title('Predicted vs Actual Frauds')
      i=i+1
      plt.savefig(f'static/images/try{i}.png')

      fig2 = plt.figure(figsize=(6,6))
      con_mat=confusion_matrix(y_test, y_test_pred)
      labels = ['No Fraud', 'Fraud']
      sns.heatmap(con_mat, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
      plt.xlabel('Predicted')
      plt.ylabel('Actual')
      i=i+1
      plt.savefig(f'static/images/try{i}.png')



@app.route('/')

@app.route('/index')
def index():
      return render_template('index.html')

@app.route('/prediction')
def prediction():
      return render_template('prediction.html')

@app.route('/login')
def login():
      return render_template('login.html')

@app.route('/analysis')
def analysis():
      return render_template('analysis.html', log_tr_re = [log_tr_re.to_html(classes="rep_tab")], log_te_re = [log_te_re.to_html(classes="rep_tab")], 
                             knn_tr_re = [knn_tr_re.to_html(classes="rep_tab")], knn_te_re = [knn_te_re.to_html(classes="rep_tab")])


@app.route('/predict',methods=['POST'])
def predict():
      feature = [int(x) for x in request.form.values()]
      out = lr.predict(np.array([feature]))
      #first_name = request.form.get("sku")
      return render_template('prediction.html', x = round(out[0]))

@app.route('/predict1',methods=['POST'])
def predict1():
      feature = [int(x) for x in request.form.values()]
      out = Kneib.predict(np.array([feature]))
      return render_template('prediction.html', x1 = round(out[0]))


if __name__ == '__main__':

      makeplot()
      



      app.run(debug=True)
