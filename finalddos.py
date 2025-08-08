import pandas as pd
import numpy as np
from scapy.all import *
from joblib import load
from joblib import dump
from decimal import Decimal


# For ploting the graphs
import matplotlib.pyplot as plt
import seaborn as sns
import csv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Machine learning Model 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Machine learning model evaluation
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, confusion_matrix


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
df=pd.read_csv("Data/DDos.csv")

df.head(10)





# Remove the spaces before the column names
df.columns = df.columns.str.strip()
df.loc[:,'Label'].unique()
#Checking the null values in the dataset.
plt.figure(1,figsize=( 10,4))
plt.hist( df.isna().sum())
# Set the title and axis labels
plt.xticks([0, 1], labels=['Not Null=0', 'Null=1'])
plt.title('Columns with Null Values')
plt.xlabel('Feature')
plt.ylabel('The number of features')

# Show the plot
plt.show()
def plotMissingValues(dataframe):
    missing_values = dataframe.isnull().sum()  # Counting null values for each column
    fig = plt.figure(figsize=(16, 5))
    missing_values.plot(kind='bar')
    plt.xlabel("Features")
    plt.ylabel("Missing values")
    plt.title("Total number of Missing values in each feature")
    plt.show()

plotMissingValues(df)

## Removing the null values
data_f=df.dropna()
#Checking the null values in the dataset.
plt.figure(1,figsize=( 10,4))
plt.hist( data_f.isna().sum())
# Set the title and axis labels
plt.title('Data aftter removing the Null Values')
plt.xlabel('The number of null values')
plt.ylabel('Number of columns')

# Show the plot
plt.show()

pd.set_option('use_inf_as_na', True)  # Treat inf as NaN
null_values=data_f.isnull().sum()  # Check for NaN values
# To know the data types of the columns

(data_f.dtypes=='object')
# Convert the labels in the DataFrame to numerical values
data_f['Label'] = data_f['Label'].map({'BENIGN': 0, 'DDoS': 1})
# Print the DataFrame

plt.hist(data_f['Label'], bins=[0, 0.3,0.7,1], edgecolor='black')  # Specify bins as [0, 1]
plt.xticks([0, 1], labels=['BENIGN=0', 'DDoS=1'])
plt.xlabel("Classes")
plt.ylabel("Count")
plt.show()
df.describe()
# Create a histogram plot for each feature
plt.figure(5)
for col in data_f.columns:
    plt.hist(data_f[col])
    plt.title(col)
    plt.show()
# Convert into numpy array

#X1=np.array(data_f).astype(np.float64)
#y1=np.array(data_f['Label'])


# Split data into features and target variable
X = data_f.drop('Label', axis=1)
y = data_f['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
print("The train dataset size = ",X_train.shape)
print("The test dataset size = ",X_test.shape)



# Random Forest
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
# Getting feature importances from the trained model
importances = rf_model.feature_importances_

# Getting the indices of features sorted by importance
indices = sorted(range(len(importances)), key=lambda i: importances[i], reverse=False)
feature_names = [f"Features {i}" for i in indices]  # Replace with your column names

# Plotting feature importances horizontally
plt.figure(figsize=(8, 14))
plt.barh(range(X_train.shape[1]), importances[indices], align="center")
plt.yticks(range(X_train.shape[1]), feature_names)
plt.xlabel("Importance")
plt.title("Feature Importances")
plt.show()
from sklearn.tree import plot_tree

estimator = rf_model.estimators_[0]  # Selecting the first estimator from the random forest model


plt.figure(figsize=(20, 10))
plot_tree(estimator, filled=True, rounded=True)
plt.show()
# Function to generate and display a detailed confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
# Evaluate Random Forest
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)

print('\nRandom Forest Metrics:')
print(f'Accuracy: {rf_accuracy:.4f}')
print(f'F1 Score: {rf_f1:.4f}')
print(f'Precision: {rf_precision:.4f}')
print(f'Recall: {rf_recall:.4f}')
# Confusion Matrix for Random Forest
plot_confusion_matrix(y_test, rf_pred, ['Benign', 'DDoS'], 'Random Forest Confusion Matrix')





lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)
print('\nLogistic Regression Metrics:')
print(f'Accuracy: {lr_accuracy:.4f}')
print(f'F1 Score: {lr_f1:.4f}')
print(f'Precision: {lr_precision:.4f}')
print(f'Recall: {lr_recall:.4f}')



# Confusion Matrix for Logistic Regression
plot_confusion_matrix(y_test, lr_pred, ['Benign', 'DDoS'], 'Logistic Regression Confusion Matrix')
nn_model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=10, random_state=42)


nn_model.fit(X_train, y_train)
nn_pred = nn_model.predict(X_test)
nn_accuracy = accuracy_score(y_test, nn_pred)
nn_f1 = f1_score(y_test, nn_pred)
nn_precision = precision_score(y_test, nn_pred)
nn_recall = recall_score(y_test, nn_pred)

print('\nNeural Network Metrics:')
print(f'Accuracy: {nn_accuracy:.4f}')
print(f'F1 Score: {nn_f1:.4f}')
print(f'Precision: {nn_precision:.4f}')
print(f'Recall: {nn_recall:.4f}')
# Confusion Matrix for Neural Network
plot_confusion_matrix(y_test, nn_pred, ['Benign', 'DDoS'], 'Neural Network Confusion Matrix')
# Random Forest
rf_proba = rf_model.predict_proba(X_test)

# Logistic Regression
lr_proba = lr_model.predict_proba(X_test)



# Neural Network
nn_proba = nn_model.predict_proba(X_test)

model = RandomForestClassifier()
model.fit(X_train, y_train)

def extract_features(packet):
    features = {
       
        'destination_port': packet[TCP].dport if TCP in packet else None,
        'flow_duration': getattr(packet, 'time', None),  # Assuming 'time' represents the flow duration
        'total_fwd_packets': calculate_total_fwd_packets(packets) if calculate_total_fwd_packets(packets) is not None else None,

        'total_bwd_packets':calculate_total_bwd_packets(packets) if calculate_total_bwd_packets(packets) is not None else None,
        'total_length_fwd_packets': calculate_total_length_fwd_packets(packets),
        'total_length_fwd_packets':calculate_total_length_fwd_packets(packets),
        'total_length_bwd_packets': calculate_total_length_bwd_packets(packets),
        'fwd_packet_length_max': calculate_fwd_packet_length_max(packets),
        'fwd_packet_length_min': calculate_fwd_packet_length_min(packets),
        'fwd_packet_length_mean':calculate_fwd_packet_length_mean(packets),
        'fwd_packet_length_std': calculate_fwd_packet_length_std(packets),
        'bwd_packet_length_max': calculate_bwd_packet_length_max(packets),
        'bwd_packet_length_min': calculate_bwd_packet_length_min(packets),
       'bwd_packet_length_min': calculate_bwd_packet_length_min(packets),
        'bwd_packet_length_mean': calculate_bwd_packet_length_mean(packets),
        'bwd_packet_length_std': calculate_bwd_packet_length_std(packets),
        'flow_bytes/s': float(calculate_flow_bytes_per_second(packets)) if calculate_flow_bytes_per_second(packets) else None,
        'flow_packets_per_second':float(calculate_flow_packets_per_second(packets)) if calculate_flow_packets_per_second(packets) else None,
        
        'flow_iat_mean': getattr(packet, 'flow_iat_mean', None),
        'flow_iat_std': getattr(packet, 'flow_iat_std', None),
        'flow_iat_max': getattr(packet, 'flow_iat_max', None),
        'flow_iat_min': getattr(packet, 'flow_iat_min', None),
        'fwd_iat_total': getattr(packet, 'fwd_iat_total', None),
        'fwd_iat_mean': getattr(packet, 'fwd_iat_mean', None),
        'fwd_iat_std': getattr(packet, 'fwd_iat_std', None),
        'fwd_iat_max': getattr(packet, 'fwd_iat_max', None),

        'fwd_iat_min': getattr(packet, 'fwd_iat_min', None),
        'bwd_iat_total': calculate_bwd_iat_total(packets),
        'bwd_iat_mean': getattr(packet, 'bwd_iat_mean', None),
        'bwd_iat_std': getattr(packet, 'bwd_iat_std', None),
        'bwd_iat_max': calculate_bwd_iat_max(packets),
        'bwd_iat_min': calculate_bwd_iat_min(packets),
        'fwd_psh_flags':calculate_fwd_psh_flags(packets),
        'bwd_psh_flags': calculate_bwd_psh_flags(packets),
        'fwd_urg_flags': calculate_fwd_urg_flags(packets),
        'bwd_urg_flags': calculate_bwd_urg_flags(packets),
        'fwd_header_length': calculate_fwd_header_length(packets),
        'bwd_header_length': calculate_bwd_header_length(packets),
        'fwd_packets/s': getattr(packet, 'fwd_packets/s', None),
        'bwd_packets/s': getattr(packet, 'bwd_packets/s', None),
        'min_packet_length': getattr(packet, 'min_packet_length', None),
        'max_packet_length': getattr(packet, 'max_packet_length', None),
        'packet_length_mean': getattr(packet, 'packet_length_mean', None),
        'packet_length_std': getattr(packet, 'packet_length_std', None),
        'packet_length_variance': getattr(packet, 'packet_length_variance', None),
        'fin_flag_count': getattr(packet, 'fin_flag_count', None),
        'syn_flag_count': getattr(packet, 'syn_flag_count', None),
        'rst_flag_count': getattr(packet, 'rst_flag_count', None),
        'psh_flag_count': getattr(packet, 'psh_flag_count', None),
        'ack_flag_count': getattr(packet, 'ack_flag_count', None),
        'urg_flag_count': getattr(packet, 'urg_flag_count', None),
        'cwe_flag_count': getattr(packet, 'cwe_flag_count', None),
        'ece_flag_count': getattr(packet, 'ece_flag_count', None),
        'down/up_ratio': getattr(packet, 'down/up_ratio', None),
        'average_packet_size': getattr(packet, 'average_packet_size', None),
        'avg_fwd_segment_size': getattr(packet, 'avg_fwd_segment_size', None),
        'avg_bwd_segment_size': getattr(packet, 'avg_bwd_segment_size', None),
        'fwd_avg_bytes/bulk': getattr(packet, 'fwd_avg_bytes/bulk', None),
        'fwd_avg_packets/bulk': getattr(packet, 'fwd_avg_packets/bulk', None),
        'fwd_avg_bulk_rate': getattr(packet, 'fwd_avg_bulk_rate', None),
        'bwd_avg_bytes/bulk': getattr(packet, 'bwd_avg_bytes/bulk', None),
        'bwd_avg_packets/bulk': getattr(packet, 'bwd_avg_packets/bulk', None),
        'bwd_avg_bulk_rate': getattr(packet, 'bwd_avg_bulk_rate', None),
        'subflow_fwd_packets': getattr(packet, 'subflow_fwd_packets', None),
        'subflow_fwd_bytes': getattr(packet, 'subflow_fwd_bytes', None),
        'subflow_bwd_packets': getattr(packet, 'subflow_bwd_packets', None),
        'subflow_bwd_bytes': getattr(packet, 'subflow_bwd_bytes', None),
        'init_win_bytes_forward': getattr(packet, 'init_win_bytes_forward', None),
        'init_win_bytes_backward': getattr(packet, 'init_win_bytes_backward', None),
        'act_data_pkt_fwd': getattr(packet, 'act_data_pkt_fwd', None),
        'min_seg_size_forward': getattr(packet, 'min_seg_size_forward', None),
        'active_mean': getattr(packet, 'active_mean', None),
        'active_std': getattr(packet, 'active_std', None),
        'active_max': getattr(packet, 'active_max', None),
        'active_min': getattr(packet, 'active_min', None),
        'idle_mean': getattr(packet, 'idle_mean', None),
        'idle_std': getattr(packet, 'idle_std', None),
        'idle_max': getattr(packet, 'idle_max', None),
        'idle_min': getattr(packet, 'idle_min', None),
        'label': getattr(packet, 'label', None)
    }
    return features


def calculate_total_fwd_packets(packets):
    return len([packet for packet in packets if TCP in packet ])
    
    
def calculate_total_bwd_packets(packets):
    return len([packet for packet in packets] )
    
def calculate_total_length_fwd_packets(packets):
    return sum([len(packet) for packet in packets] )
    
def calculate_total_length_bwd_packets(packets):
    return sum([len(packet) for packet in packets])
    
def calculate_fwd_packet_length_max(packets):
    max_length = 0
    for packet in packets:
        if TCP in packet and packet[TCP].sport == 80:
            length = len(packet)
            max_length = max(max_length, length)
    return max_length
    
def calculate_fwd_packet_length_min(packets):
    min_length = float('inf')
    for packet in packets:
        if TCP in packet and packet[TCP].sport == 80:
            length = len(packet)
            min_length = min(min_length, length)
    return min_length if min_length != float('inf') else None
    
def calculate_fwd_packet_length_mean(packets):
    total_length = 0
    num_packets = 0
    for packet in packets:
       
        total_length += len(packet)
        num_packets += 1
    return total_length / num_packets if num_packets > 0 else None

def calculate_fwd_packet_length_std(packets):
    fwd_packet_lengths = [len(packet) for packet in packets ]
    return np.std(fwd_packet_lengths) if fwd_packet_lengths else None

def calculate_bwd_packet_length_max(packets):
    bwd_packet_lengths = [len(packet) for packet in packets ]
    return max(bwd_packet_lengths) if bwd_packet_lengths else None

def calculate_bwd_packet_length_min(packets):
    bwd_packet_lengths = [len(packet) for packet in packets ]
    return min(bwd_packet_lengths) if bwd_packet_lengths else None
    
def calculate_bwd_packet_length_mean(packets):
    bwd_packet_lengths = [len(packet) for packet in packets ]
    return np.mean(bwd_packet_lengths) if bwd_packet_lengths else None

def calculate_bwd_packet_length_std(packets):
    bwd_packet_lengths = [len(packet) for packet in packets ]
    return np.std(bwd_packet_lengths) if bwd_packet_lengths else None
    
def calculate_flow_bytes_per_second(packets):
    total_bytes = sum([len(packet) for packet in packets])
    start_time = packets[0].time
    end_time = packets[-1].time
    flow_duration = end_time - start_time
    return total_bytes / flow_duration if flow_duration > 0 else None

def calculate_flow_packets_per_second(packets):
    num_packets = len(packets)
    start_time = packets[0].time
    end_time = packets[-1].time
    flow_duration = end_time - start_time
    return num_packets / flow_duration if flow_duration > 0 else None

def calculate_flow_iat_mean(packets):
    iats = [packets[i + 1].time - packet.time for i, packet in enumerate(packets[:-1])]
    return np.mean(iats) 

def calculate_flow_iat_std(packets):
    iats = [packets[i + 1].time - packet.time for i, packet in enumerate(packets[:-1])]
    return np.std(iats)

def calculate_flow_iat_max(packets):
    iats = [packets[i + 1].time - packet.time for i, packet in enumerate(packets[:-1])]
    return max(iats) 

def calculate_flow_iat_min(packets):
    iats = [packets[i + 1].time - packet.time for i, packet in enumerate(packets[:-1])]
    return min(iats) 

def calculate_fwd_iat_total(packets):
    return sum([packet.time - packets[0].time for packet in packets])

def calculate_fwd_iat_mean(packets):
    iats = [packet.time - packets[i - 1].time for i, packet in enumerate(packets[1:], start=1)]
    return np.mean(iats)
    
def calculate_fwd_iat_std(packets):
    iats = [packet.time - packets[i - 1].time for i, packet in enumerate(packets[1:], start=1)]
    return np.std(iats)

def calculate_fwd_iat_max(packets):
    iats = [packet.time - packets[i - 1].time for i, packet in enumerate(packets[1:], start=1)]
    return max(iats) 

def calculate_fwd_iat_min(packets):
    iats = [packet.time - packets[i - 1].time for i, packet in enumerate(packets[1:], start=1)]
    return min(iats) 

def calculate_bwd_iat_total(packets):
    return sum([packet.time - packets[i - 1].time for i, packet in enumerate(packets[1:], start=1)])

def calculate_bwd_iat_mean(packets):
    iats = [packet.time - packets[i - 1].time for i, packet in enumerate(packets[1:], start=1)]
    return np.mean(iats) if iats else None

def calculate_bwd_iat_std(packets):
    iats = [packet.time - packets[i - 1].time for i, packet in enumerate(packets[1:], start=1)]
    return np.std(iats) if iats else None

def calculate_bwd_iat_max(packets):
    iats = [packet.time - packets[i - 1].time for i, packet in enumerate(packets[1:], start=1)]
    return max(iats) if iats else None

def calculate_bwd_iat_min(packets):
    iats = [packet.time - packets[i - 1].time for i, packet in enumerate(packets[1:], start=1)]
    return min(iats) if iats else None

def calculate_fwd_psh_flags(packets):
    return sum([1 for packet in packets if packet[TCP].flags.PSH]) 

def calculate_bwd_psh_flags(packets):
    return sum([1 for packet in packets if packet[TCP].flags.PSH]) 

def calculate_fwd_urg_flags(packets):
    return sum([1 for packet in packets if packet[TCP].flags.URG]) 

def calculate_bwd_urg_flags(packets):
    return sum([1 for packet in packets if packet[TCP].flags.URG]) 

def calculate_fwd_header_length(packets):
    return sum([len(packet[TCP].payload) for packet in packets]) 

def calculate_bwd_header_length(packets):
    return sum([len(packet[TCP].payload) for packet in packets]) 

def calculate_fwd_packets_per_second(packets):
    flow_duration = calculate_flow_duration(packets)
    return len(packets) / flow_duration if flow_duration else None
    
def calculate_flow_duration(packets):
    return (packets.time())
    

def calculate_bwd_packets_per_second(packets):
    flow_duration = calculate_flow_duration(packets)
    return len(packets) / flow_duration if flow_duration else None

def calculate_min_packet_length(packets):
    return min([len(packet[TCP].payload) for packet in packets]) 

def calculate_max_packet_length(packets):
    return max([len(packet[TCP].payload) for packet in packets]) 

def calculate_packet_length_mean(packets):
    lengths = [len(packet[TCP].payload) for packet in packets ]
    return decimal(np.mean(lengths)) if lengths else None
    
def calculate_packet_length_std(packets):
    lengths =[len(packet[TCP].payload) for packet in packets ]
    return decimal(np.std(lengths)) if lengths else None

def calculate_packet_length_variance(packets):
    lengths = [len(packet[TCP].payload) for packet in packets]
    return decimal(np.var(lengths)) if lengths else None

def calculate_fin_flag_count(packets):
    return sum([1 for packet in packets if TCP in packet and packet[TCP].flags.FIN]) if TCP in packets else None

def calculate_syn_flag_count(packets):
    return sum([1 for packet in packets if TCP in packet and packet[TCP].flags.SYN]) if TCP in packets else None

def calculate_rst_flag_count(packets):
    return sum([1 for packet in packets if TCP in packet and packet[TCP].flags.RST]) if TCP in packets else None

def calculate_psh_flag_count(packets):
    return sum([1 for packet in packets if TCP in packet and packet[TCP].flags.PSH]) if TCP in packets else None

def calculate_ack_flag_count(packets):
    return sum([1 for packet in packets if TCP in packet and packet[TCP].flags.ACK]) if TCP in packets else None

def calculate_urg_flag_count(packets):
    return sum([1 for packet in packets if TCP in packet and packet[TCP].flags.URG]) if TCP in packets else None

def calculate_cwe_flag_count(packets):
    return sum([1 for packet in packets if TCP in packet and packet[TCP].flags.CWR]) if TCP in packets else None

def calculate_ece_flag_count(packets):
    return sum([1 for packet in packets if TCP in packet and packet[TCP].flags.ECE]) if TCP in packets else None

def calculate_down_up_ratio(packets):
    down_packets = len([1 for packet in packets ]) if TCP in packets else 0
    up_packets = len(packets) - down_packets
    return down_packets / up_packets if up_packets != 0 else None

def calculate_average_packet_size(packets):
    lengths = [len(packet[TCP].payload) for packet in packets ]
    return sum(lengths) / len(packets) if lengths else None

def calculate_avg_fwd_segment_size(packets):
    lengths = [len(packet[TCP].payload) for packet in packets ]
    return sum(lengths) / len(lengths) if lengths else None


def calculate_avg_bwd_segment_size(packets):
    lengths = [len(packet[TCP].payload) for packet in packets]
    return sum(lengths) / len(packets) if lengths else None


def calculate_subflow_fwd_packets(packets):
    return len([packet for packet in packets]) 

def calculate_subflow_fwd_bytes(packets):
    return sum([len(packet[TCP].payload) for packet in packets ]) 

def calculate_subflow_bwd_packets(packets):
    return len([packet for packet in packets ]) 

def calculate_subflow_bwd_bytes(packets):
    return sum([len(packet[TCP].payload) for packet in packets ]) 

def calculate_label(features):
    # Assuming you have some logic to determine if the flow is malicious based on the features
    if features['fwd_packets/s'] > 10 and features['bwd_packets/s'] > 10:
        return '1'
    else:
        return '0'


# Define a function to predict DDoS attacks based on extracted features
def predict(features):
   # print(features)
    df = pd.DataFrame([features])
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    prediction = model.predict(df_scaled)
    return prediction[0]

def packet_callback(packet):
    try:
            #if TCP in packet:
                # Check if the packet has an IP layer
                # Extract features from the packet
            features = extract_features(packet)
        
                # Make a prediction
            prediction = predict(features)
            if prediction == 1:  # Assuming 1 indicates a DDoS attack in your model
                    # Mitigation action (e.g., drop packet, alert, etc.)
                print(f"Potential DDoS attack detected from {packet[IP].src}")
            else:
                    # Allow the packet to pass
                print(f"Packet allows from {packet[IP].src}")
                send(packet, verbose=0)
            
    except Exception as e:
        print(f"Error processing packet: {e}")

# Capture packets and apply the callback function
captured_packets = sniff(filter='tcp', store=1, count=10)

# Write the captured packets to a PCAP file
wrpcap('captured_packets.pcap', captured_packets)

# Open the PCAP file for reading
pcap_file = "captured_packets.pcap"
packets = rdpcap(pcap_file)

# Iterate over each packet and process it
for packet in packets:
    packet_callback(packet)
%history -f finalddos.py
