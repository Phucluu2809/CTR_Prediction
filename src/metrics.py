from sklearn.metrics import roc_auc_score, log_loss


def compute_auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)


def compute_logloss(y_true, y_pred):
    return log_loss(y_true, y_pred)