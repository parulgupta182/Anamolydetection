import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import pandas as pd
from matplotlib import pyplot as plt
import datetime
import pyshark
import asyncio
from pyshark.tshark.tshark import get_tshark_interfaces, get_process_path
import os
import subprocess
from sklearn import metrics


def list_interfaces():
    parameters = [get_process_path(), "-D"]
    with open(os.devnull, "w") as null:
        tshark_interfaces = subprocess.check_output(parameters, stderr=null).decode("utf-8")

    for line in tshark_interfaces.splitlines():
        print(line.split(" ")[1])
        print(line.split(".")[0])


def ProcessPackets(packet):
    packet_version = packet.layers[1].version
    layer_name = packet.layers[2].layer_name
    packet_list.append([packet_version, layer_name, packet.length, packet.sniff_time])


def Capture():
    global packet_list
    packet_list = []
    timeout = 5
    # capture = pyshark.LiveCapture(interface=self.interface, decode_as={"tcp.port==4713": "synphasor"})
    capture = pyshark.LiveCapture(interface='lo0', capture_filter='tcp')
    capture.load_packets()
    capture.apply_on_packets(ProcessPackets)
    data = pd.DataFrame(packet_list, columns=['vIP', 'protocol', 'length', 'timestamp'])
    print(data['timestamp'].iloc[-1] - data['timestamp'].iloc[0])


def feature_extraction(cap):
    drn_time = []
    pkt_length = []
    lyr = []
    payload_len = []
    time = []
    time_array = []
    for pcap in cap:
        payload_length = 0
        has_layer = pcap.layers is not None
        layer = pcap.layers[-1]
        timestamp = float(pcap.sniff_timestamp)
        duration_time = str(pcap.sniff_time)
        packet_length = int(pcap.length)
        if 'tcp' in pcap:
            field_names = pcap.tcp._all_fields
            field_values = pcap.tcp._all_fields.values()
            for key in field_names:
                for field_value in field_values:
                    if 'tcp.payload' in key:
                        payload_length = len(field_value)

        if has_layer:
            # pd.DataFrame([layer, packet_time, duration_time, timestamp,  packet_length, payload_length]).to_csv('Features.csv',mode='a')
            time.append(timestamp)
            drn_time.append(duration_time)
            pkt_length.append(packet_length)
            lyr.append(layer)
            payload_len.append(payload_length)

    i = 1
    for i in range(len(time)):
        time_array.append(time[i] - time[i - 1])

    time_array.pop(0)
    pkt_length.pop(0)
    payload_len.pop(0)
    dict_layer = {'layer': lyr}
    df1 = pd.DataFrame(dict_layer)

    dict = {'time_difference': time_array, 'packet_length': pkt_length, 'payload_length': payload_len}
    df = pd.DataFrame(dict)
    return df, df1


def svdd():
    # Preprocess and fit PCA
    data = pd.read_csv("Features_train_new.csv")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)  # transform data
    pca = PCA(n_components=2)  # PCA
    pca.fit(scaled_data)

    # Split data into training / testing
    train_data, validation_data = train_test_split(scaled_data, train_size=0.8)
    train_data_p = pca.transform(train_data)
    validation_data_p = pca.transform(validation_data)

    test_data_raw = pd.read_csv("Features_test_new.csv")
    test_data = scaler.fit_transform(test_data_raw)  # transform data
    pca = PCA(n_components=2)  # PCA
    pca.fit(scaled_data)
    test_data_p = pca.transform(test_data)

    # Train One-Class SVM
    ocsvm = OneClassSVM(kernel='linear', gamma='auto', tol=1e-3, nu=0.02, shrinking=True, max_iter=-1).fit(train_data)
    ocsvm.fit(train_data)
    pred_tr = ocsvm.predict(train_data)
    pred_test = ocsvm.predict(test_data)

    # print("Accuracy:", metrics.accuracy_score(test_data, pred_test))
    # print("Precision:",metrics.precision_score(test_data, pred_test))

    print("shape of ocsvm output ", pred_tr.shape)
    score = ocsvm.score_samples(train_data)
    # print(score)

    dict = {'score_samples': score}
    df = pd.DataFrame(dict)
    df.to_csv('score_samples.csv')

    # Count outliers / anomalies
    outliers_tr = np.where(pred_tr == -1)[0]
    inliers_tr = np.where(pred_tr == 1)[0]
    outliers_test = np.where(pred_test == -1)[0]
    inliers_test = np.where(pred_test == 1)[0]
    print("Fraction of outliers (training)", np.size(outliers_tr) / np.size(pred_tr))
    print("Fraction of outliers (test)", np.size(outliers_test) / np.size(pred_test))

    # 2D-Projection results (training)
    plt.figure()
    colors = ['navy', 'red']
    plt.scatter(train_data_p[inliers_tr, 0], train_data_p[inliers_tr, 1],
                color=colors[0], alpha=.8, lw=2, label='Inlier')
    plt.scatter(train_data_p[outliers_tr, 0], train_data_p[outliers_tr, 1],
                color=colors[1], alpha=.8, lw=2, label='Outlier')
    plt.xlabel('Component 1 (' + str(np.around(100 * pca.explained_variance_ratio_[0], 0)) + ' %)')
    plt.ylabel('Component 2 (' + str(np.around(100 * pca.explained_variance_ratio_[1], 0)) + ' %)')
    plt.legend()
    plt.show()

    # 2D-Projection results (testing)
    plt.figure()
    colors = ['navy', 'red']
    plt.scatter(test_data_p[inliers_test, 0], test_data_p[inliers_test, 1],
                color=colors[0], alpha=.8, lw=2, label='Inlier')
    plt.scatter(test_data_p[outliers_test, 0], test_data_p[outliers_test, 1],
                color=colors[1], alpha=.8, lw=2, label='Outlier')
    plt.xlabel('Component 1 (' + str(np.around(100 * pca.explained_variance_ratio_[0], 0)) + ' %)')
    plt.ylabel('Component 2 (' + str(np.around(100 * pca.explained_variance_ratio_[1], 0)) + ' %)')
    plt.legend()
    plt.show()


def step1():
    cap_train = pyshark.FileCapture('New PCAPs/PCAPs 21-07-08/21-07-08 PMU Baseline.pcap')
    #cap_train = pyshark.FileCapture('New PCAPs/21-03-17 PMU Baseline.pcap')

    df, df1 = feature_extraction(cap_train)
    df.to_csv('Features_train_new.csv')
    df1.to_csv('Features_train_layer')

    # dict1 = {'Layer': lyr}
    # df = pd.DataFrame(dict1)
    # df.to_csv('Features3.csv')

    #cap_test = pyshark.FileCapture('New PCAPs/PCAPs 21-07-08/21-07-08 PMU MITM Attack Phasor Manipulation.pcap')
    cap_test = pyshark.FileCapture('New PCAPs/PCAPs 21-07-08/21-07-08 PMU MITM Attack Freq Manipulation.pcap')
    df2, df3 = feature_extraction(cap_test)
    df2.to_csv('Features_test_new.csv')


if __name__ == '__main__':
    # list_interfaces()
    # Capture()
    step1()
    svdd()
