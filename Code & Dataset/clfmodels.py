from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
#import xgboost as xgb

def GaussianNaiveBayes(x,y):
    mod = GaussianNB()
    mod.fit(x,y)
    result = {}
    result["Name"] = "Gaussian Naive Bayes Classifier"
    result["Model"] = mod
    result["Train Data - Accuracy/Score"] = mod.score(x,y)
    return(result)

def MultinomialNaiveBayes(x,y):
    mod = MultinomialNB()
    mod.fit(x,y)
    result = {}
    result["Name"] = "Multinomial Naive Bayes Classifier"
    result["Model"] = mod
    result["Train Data - Accuracy/Score"] = mod.score(x,y)
    return(result)

def SVMClassifier(x,y):
    mod = SVC(kernel='linear')
    mod.fit(x,y)
    result = {}
    result["Name"] = "Support Vector Machine Classfiier"
    result["Model"] = mod
    result["Train Data - Accuracy/Score"] = mod.score(x,y)
    return(result)

def CART(x,y):
    mod = DecisionTreeClassifier()
    mod.fit(x,y)
    result = {}
    result["Name"] = "Classification & Regression Tree"
    result["Model"] = mod
    result["Train Data - Accuracy/Score"] = mod.score(x,y)
    return(result)

def ID3(x,y):
    mod = DecisionTreeClassifier(criterion='entropy')
    mod.fit(x,y)
    result = {}
    result["Name"] = "ID3 Classifier"
    result["Model"] = mod
    result["Train Data - Accuracy/Score"] = mod.score(x,y)
    return(result)

def C4_5(x,y):
    mod = DecisionTreeClassifier(criterion='entropy',splitter='best')
    mod.fit(x,y)
    result = {}
    result["Name"] = "C4.5 Classifier"
    result["Model"] = mod
    result["Train Data - Accuracy/Score"] = mod.score(x,y)
    return(result)

def RandomForest(x,y):
    mod = RandomForestClassifier()
    mod.fit(x,y)
    result = {}
    result["Name"] = "Random Forest Classifier"
    result["Model"] = mod
    result["Train Data - Accuracy/Score"] = mod.score(x,y)
    return(result)

def GradientBooster(x,y):
    mod = GradientBoostingClassifier()
    mod.fit(x,y)
    result = {}
    result["Name"] = "Gradient Boost Classifier"
    result["Model"] = mod
    result["Train Data - Accuracy/Score"] = mod.score(x,y)
    return(result)

def AdaBoost(x,y):
    mod = AdaBoostClassifier()
    mod.fit(x,y)
    result = {}
    result["Name"] = "Ada Boost Classifier"
    result["Model"] = mod
    result["Train Data - Accuracy/Score"] = mod.score(x,y)
    return(result)

"""def XGBoost(x,y):
    mod = xgb.XGBClassifier()
    mod.fit(x,y)
    result = {}
    result["Name"] = "XGBoost Classifier"
    result["Model"] = mod
    result["Train Data - Accuracy/Score"] = mod.score(x,y)
    return(result)"""