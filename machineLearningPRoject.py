##################################################
######## designing the GUI using TKINTER #########

from tkinter import *
import tkinter.messagebox as m
w=Tk()
w.title("IRIS flower classification")
w.configure(bg='coral2')
w.geometry('650x400')



import warnings
warnings.simplefilter(action='ignore')

import matplotlib.pyplot as plt


from sklearn.datasets import load_iris ##importing the data of iris flower
iris=load_iris()
X=iris.data ##numpy array
Y=iris.target  ##numpy array
##split dataset for training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
acc_knn=0
acc_knn1=0
acc_knn2=0
acc_knn4=0



###CREATE A MODEL  ## KNN
##K Nearest Neighbors algorithm

from sklearn.neighbors import KNeighborsClassifier
K=KNeighborsClassifier(n_neighbors=5)
  
##Train the model BY training dataset
K.fit(X_train,Y_train)
 
##test the model by testing data
Y_pred=K.predict(X_test)

##find accuracy
from sklearn.metrics import accuracy_score
acc_knn=accuracy_score(Y_test,Y_pred)
acc_knn=round(acc_knn*100,2)

def knn():
    c=str(acc_knn) + " %"
    m.showinfo(title="KNN",message=c)    
    print("accuracy score in KNN is",acc_knn,"%")
    print(K.predict([[6,4,3,4]]))

    
def lg():
    c=str(acc_knn1) + " %"
    m.showinfo(title="LR",message=c)
    print("accuracy score in LG is",acc_knn1,"%")
    print(logreg.predict([[6,4,3,4]]))
    
##logistic regression
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X_train,Y_train)
Y_pred1=logreg.predict(X_test)



from sklearn.metrics import accuracy_score
acc_knn1=accuracy_score(Y_test,Y_pred1)
acc_knn1=round(acc_knn1*100,2)


def dt():
    c="gini: "+ str(acc_knn2) + " %"
    c=c + " entropy: "+ str(acc_knn3) + " %"
    m.showinfo(title="DT ENTROPY",message=c)
    print("accuracy score in DT GINI is",acc_knn2,"%")
    print(dtgini.predict([[6,4,3,4]]))
    print("accuracy score in DT ENTROPY is",acc_knn3,"%")
    print(dtentropy.predict([[6,4,3,4]]))

    
###Decision Tree
from sklearn.tree import DecisionTreeClassifier
    
dtgini=DecisionTreeClassifier(criterion="gini")
dtgini.fit(X_train,Y_train)
Y_pred2=dtgini.predict(X_test)
from sklearn.metrics import accuracy_score
acc_knn2=accuracy_score(Y_test,Y_pred2)
acc_knn2=round(acc_knn2*100,2)
    
dtentropy=DecisionTreeClassifier(criterion="entropy")
dtentropy.fit(X_train,Y_train)
Y_pred3=dtentropy.predict(X_test)
from sklearn.metrics import accuracy_score
acc_knn3=accuracy_score(Y_test,Y_pred3)
acc_knn3=round(acc_knn3*100,2)

if acc_knn3>acc_knn2:
    acc_knndt=acc_knn3
else:
    acc_knndt=acc_knn2


    
def nb():
    c= str(acc_knn4) + " %"
    m.showinfo(title="NB",message=c)
    print("accuracy score in NB is",acc_knn4,"%")
    print(nbayes.predict([[6,4,3,4]]))
    

##naive bayes
from sklearn.naive_bayes import GaussianNB
nbayes=GaussianNB()
nbayes.fit(X_train,Y_train)
Y_pred4=nbayes.predict(X_test)
from sklearn.metrics import accuracy_score
acc_knn4=accuracy_score(Y_test,Y_pred4)
acc_knn4=round(acc_knn4*100,2)


def display(a):
    if a[0]==0:
        s5.set("SETOSA")
    elif a[0]==1:
        s5.set("VERSICOLOR")
    else:
        s5.set("VERGINICA")



def submit():
    a=s1.get()
    b=s2.get()
    c=s3.get()
    d=s4.get()
    
    if acc_knn>=acc_knn1 and (acc_knn>=acc_knndt and acc_knn>=acc_knn4):
        print("KNN: ",K.predict([[a,b,c,d]]))
        display(K.predict([[a,b,c,d]]))
    elif acc_knn1>=acc_knndt and acc_knn1>=acc_knn4:
        print("LG: ",logreg.predict([[a,b,c,d]]))
        display(logreg.predict([[a,b,c,d]]))
    elif acc_knndt>=acc_knn4:
        if acc_knn3>=acc_knn2:
            print("DT GINI: ",dtgini.predict([[a,b,c,d]]))
            display(dtgini.predict([[a,b,c,d]]))
        else:
            print("DT ENTROPY: ",dtentropy.predict([[a,b,c,d]]))
            display(dtentropy.predict([[a,b,c,d]]))
    else:
        print("NB: ",nbayes.predict([[a,b,c,d]]))
        display(nbayes.predict([[a,b,c,d]]))



def reset():
    s1.set("")
    s2.set("")
    s3.set("")
    s4.set("")
    s5.set("Enter Data")

def compare():
    bottom=0
    # x-cordinate
    left=[1,2,3,4]

    #y-coordinate
    height=[acc_knn,acc_knn1,acc_knndt,acc_knn4]

    #label bar
    tick_label=['KNN','LR,','DT','NB']

    #plotting graph
    plt.bar(left,height,tick_label=tick_label,width=0.8,color=['red','green'])

    #labeling axis
    plt.xlabel=('MODEL')
    plt.ylabel=('ACCURACY')
    plt.title('FLOWER ML')
    plt.show()









        






s1=IntVar()
s2=IntVar()
s3=IntVar()
s4=IntVar()
s5=StringVar()





####### designing the component of the GUI #######
l2=Label(w, text="                     ", bg="coral2")
l2.grid(row=0, column=1)
l=Label(w, text="'''        Enter Following File        '''", fg="firebrick4" ,font=("Times New Roman",15,"underline"), bg="coral2")
l.grid(row=1, column=3, columnspan=2)
l1=Label(w, text="            ", bg="coral2")
l1.grid(row=2, column=0)



####left side UI
knn=Button(w, text="KNN", font=("arial", 15, "bold"), height=1,width=6, bg="firebrick4", command=knn)
knn.grid(row=2, column=1)
l3=Label(w, text="       ", bg="coral2")
l3.grid(row=3, column=1)


lg=Button(w, text="LG", font=("arial", 15, "bold"), height=1,width=6, bg="firebrick4", command=lg)
lg.grid(row=4, column=1)
l4=Label(w, text="       ", bg="coral2")
l4.grid(row=5, column=1)

dt=Button(w, text="DT", font=("arial", 15, "bold"), height=1,width=6, bg="firebrick4",command=dt)
dt.grid(row=6, column=1)
l5=Label(w, text="       ", bg="coral2")
l5.grid(row=7, column=1)

nb=Button(w, text="NB", font=("arial", 15, "bold"), height=1,width=6, bg="firebrick4",command=nb)
nb.grid(row=8, column=1)
l6=Label(w, text="       ", bg="coral2")
l6.grid(row=9, column=1)

comp=Button(w, text="Compare", font=("arial", 15, "bold"), command=compare,height=1,width=6, bg="firebrick4")
comp.grid(row=10,column=1)
l7=Label(w, text="       ", bg="coral2")
l7.grid(row=11, column=1)


l8=Label(w, text="                                 ", bg="coral2")
l8.grid(row=2, column=2)
result=Label(w, font=("arial", 20, "bold"), textvariable=s5, relief="solid", bg="coral4", width=35)
result.grid(row=12, column=1, columnspan=6)
####right side UI
sl=Label(w, text="SL  ", bg="coral2", font=("arial", 15, "bold"), height=1,width=6, justify="left")
sl.grid(row=2, column=3)
slEntry=Entry(w, font=("arial", 15, "bold"), relief="solid", textvariable=s1)
slEntry.grid(row=2, column=4)

sw=Label(w, text="SW  ", bg="coral2", font=("arial", 15, "bold"), height=1,width=6)
sw.grid(row=4, column=3)
swEntry=Entry(w, font=("arial", 15, "bold"), relief="solid", textvariable=s2)
swEntry.grid(row=4, column=4)

pl=Label(w, text="PL  ", bg="coral2", font=("arial", 15, "bold"), height=1,width=6)
pl.grid(row=6, column=3)
plEntry=Entry(w, font=("arial", 15, "bold"), relief="solid", textvariable=s3)
plEntry.grid(row=6, column=4)

pw=Label(w, text="PW  ", bg="coral2", font=("arial", 15, "bold"), height=1,width=6)
pw.grid(row=8, column=3)
pwEntry=Entry(w, font=("arial", 15, "bold"), relief="solid", textvariable=s4)
pwEntry.grid(row=8, column=4)

submit=Button(w, text="Submit", font=("arial", 15, "bold"), height=1, width=6, bg="firebrick4", command=submit)
submit.grid(row=10, column=3, columnspan=1)
reset=Button(w, text="Reset", font=("arial", 15, "bold"), height=1, width=6, bg="firebrick4", command=reset)
reset.grid(row=10, column=4, columnspan=1)

w.mainloop()
######################################################
