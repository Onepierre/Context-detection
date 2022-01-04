from sklearn.metrics import f1_score,accuracy_score

# Basics accuracy and F1
def scores(y_true, y_pred):
    print(y_true,y_pred)
    return (accuracy_score(y_true, y_pred) ,f1_score(y_true, y_pred,average='weighted'))