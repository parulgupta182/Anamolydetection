import numpy as np
import pyshark
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import pandas as pd
from matplotlib import pyplot as plt

def feature_extraction():
    cap = pyshark.FileCapture('Test.pcap')

    drn_time=[]
    pkt_length=[]
    lyr=[]
    payload_len=[]
    time=[]

    for pcap in cap:

        has_transport = pcap.transport_layer is not None
        layer = pcap.transport_layer.upper()
        timestamp = float(pcap.sniff_timestamp)
        duration_time = str(pcap.sniff_time)
        packet_length = int(pcap.length)

        field_names = pcap.tcp._all_fields
        field_values = pcap.tcp._all_fields.values()
        for field_name in field_names:
            for field_value in field_values:
                if field_name == 'tcp.payload':
                    payload_length = len(field_value)



        if has_transport:

            #pd.DataFrame([layer, packet_time, duration_time, timestamp,  packet_length, payload_length]).to_csv('Features.csv',mode='a')
            time.append(timestamp)
            drn_time.append(duration_time)
            pkt_length.append(packet_length)
            lyr.append(layer)
            payload_len.append(payload_length)



    dict = { 'timestamp': time, 'packet_length': pkt_length,'payload_length': payload_len}
    #dict = {'Layer': lyr}
    df = pd.DataFrame(dict)


    df.to_csv('Features1.csv')


def svdd():

    data = pd.read_csv("Features1.csv")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)  # transform data
    pca = PCA(n_components=2)  # PCA
    projected = pca.fit_transform(scaled_data)


    print(pca.explained_variance_ratio_)
    targets = ['Outlier', 'Normal']
    plt.figure()
    colors = ['navy', 'red']
    for color, target_name in zip(colors, targets):
        plt.scatter(projected[:, 0], projected[:, 1], color=color, alpha=.5, lw=2,
                    label=target_name)
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.legend()
    plt.show()

    train_data, test_data = train_test_split(scaled_data, train_size=0.8)
    ocsvm = OneClassSVM(kernel='rbf', gamma='auto', tol=1e-3, nu=0.05, shrinking=True, max_iter=-1).fit(train_data)
    ocsvm.fit(train_data)
    pred = ocsvm.predict(train_data)

    print("shape of ocsvm output ", pred.shape)

    mask = pred != -1
    X_train, y_train = train_data[mask, :], train_data[mask]
    print(X_train.shape, y_train.shape)

    # res=pred.reshape(198,2)
    # res = pred.reshape(792, 2)

    count = 0
    for x in np.nditer(pred):
        if x == -1:
            count = count + 1

    print("number of outliers", count)


if __name__ == '__main__':
    #feature_extraction()
    svdd()





