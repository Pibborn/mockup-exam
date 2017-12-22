from utils import load_data

if __name__ == '__main__':
    kmeans = KMeans(2)
    model_list = [kmeans]
    all_data = load_data('breast-cancer-wisconsin.data')
    # your work here...
