from sklearn.metrics import F1_score,accuracy_score

# Basics accuracy and F1
def scores(y_true, y_pred):
    reutrn (accuracy_score(y_true, y_pred) ,F1_score(y_true, y_pred))