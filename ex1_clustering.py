from utils import load_data
###
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import numpy as np
from sklearn.preprocessing import Imputer, normalize
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt
###

###
def preprocess_data(data):
    # handle missing values;
    # scale the data
    imp = Imputer()
    imp.fit_transform(data)
    data = imp.transform(data)
    data = normalize(data)
    return data
###

###
def clean_labels(labels):
    # 2 -> 0; 4 -> 1
    temp = []
    for l in labels:
        if l == 2:
            temp.append(0)
        else:
            temp.append(1)
    return np.array(temp)
###

def report_nmi(data, target, model_list):
    # your work here...
    pass
    ###
    nmi_list = []
    for model in model_list:
        y_pred = model.fit_predict(data)
        nmi_list.append(normalized_mutual_info_score(y_pred, target))

    return nmi_list    
    ###

if __name__ == '__main__':
    kmeans = KMeans(2)
    model_list = [kmeans]

    # your work here...

    ###
    all_data = load_data('breast-cancer-wisconsin.data')
    data = all_data[:, 1:-1] # discarding the id
    target = all_data[:, -1]
    target = clean_labels(target)
    data = preprocess_data(data)
    _, X_val, _, y_val = train_test_split(data, target, test_size=0.2)

    agglo = AgglomerativeClustering(2)
    agglo_cosine = AgglomerativeClustering(2, affinity='cosine', linkage='average')
    agglo_cosine_2 = AgglomerativeClustering(2, affinity='cosine', linkage='complete')
    dbscan_bad = DBSCAN(eps=0.5)
    dbscan_good = DBSCAN(eps=0.1)
    model_list = [kmeans, agglo, agglo_cosine, agglo_cosine_2, dbscan_bad, dbscan_good]

    nmi_list = report_nmi(data, target, model_list)
    plt.plot(nmi_list)
    print(nmi_list)

    params = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
    best = 0
    best_eps = 0.01
    for p in params:
        model_list = [DBSCAN(eps=p)]
        dbscan_nmi_list = report_nmi(X_val, y_val, model_list)
        print(dbscan_nmi_list[0])
        if dbscan_nmi_list[0] > best:
            best = dbscan_nmi_list[0]
            best_eps = p
    y_pred_best = DBSCAN(eps=best_eps).fit_predict(data)
    print(normalized_mutual_info_score(y_pred_best, target))

    plt.show()
    ###

