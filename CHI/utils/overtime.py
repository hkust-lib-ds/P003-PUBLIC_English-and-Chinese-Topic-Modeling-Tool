import random
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import zipfile
import plotly.graph_objects as go
import io
from plotly.subplots import make_subplots


'''
colors = [
    "#8A9B9B", "#7A9B8A", "#A1B3B2", "#8E9497", "#B3B8B6", "#7D8F8D",
    "#B5B8B4", "#A59A8B", "#B8B0A7", "#7A7A7A", "#A79BB2", "#8A9B9E",
    "#C1A59A", "#A89E8D", "#8F7B6C", "#B1A89D", "#7D8F8A", "#B8A8A8",
    "#A7A79C", "#7F8C8A", "#B1A1A1", "#B69C9C", "#7DAF9E", "#8BAFAD",
    "#C6A6B1", "#A8BFB1", "#98B1A8", "#B39EB1", "#A79BCE", "#A4B6B4"
]
'''

colors = ["#0056d6", "#3571dc", "#5689e5", "#739eee", "#8eb2f6", "#a8c6fe",
          "#669c35", "#7aac50", "#8fbc69", "#a3cb84", "#b8da9c", "#cce8b5",
          "#ff6a00", "#ff8a22", "#ffa440", "#ffbc5f", "#fed083", "#ffe4a8",
          "#ff6251", "#ff7696", "#ff8ac0", "#ff9ddd", "#f8b3f2", "#f1c9fe",
          "#00a3d7", "#00b5c8", "#00c8ac", "#52d66d", "#b3d336", "#fec700"
          ]

def rescale_time(training_docs):
    all_years = [training_docs[doc]['time'][0] if type(training_docs[doc]['time'][0]) == int else 0 for doc in training_docs]
    #st.write(all_years)
    minYear = min(all_years)
    Doc_time = {}
    for docID in training_docs:
        Doc_time[docID] = {}
        year = training_docs[docID]['time'][0]
        if type(year) != int:
            year = 0
        month = training_docs[docID]['time'][1]
        if type(month) != int:
            month = 0
        day = training_docs[docID]['time'][2]
        if type(day) != int:
            day = 0
        scaledTime = (year - minYear) * 365 + month * 30 + day
        Doc_time[docID]['scaledTime'] = scaledTime
    return Doc_time
    

def GenerateTimestamp(training_docs, cluster_K):
    doc_time = rescale_time(training_docs)
    

    times = np.array([[doc_time[docID]['scaledTime']] for docID in doc_time])
    #st.write(times)

    kmeans = KMeans(n_clusters=cluster_K, random_state=0).fit(times)
    labels = kmeans.labels_

    for i, docID in enumerate(doc_time):
        training_docs[docID]['timestamp'] = labels[i]

    # reorder the cluster by the mean of time
    cluster_time = {}
    for i in range(cluster_K):
        cluster_time[i] = []
    for docID in doc_time:
        cluster_time[training_docs[docID]['timestamp']].append(doc_time[docID]['scaledTime'])
    for i in range(cluster_K):
        cluster_time[i] = np.mean(cluster_time[i])
    cluster_time = dict(sorted(cluster_time.items(), key=lambda item: item[1]))
    # reorder mapping timestamp -> cludter_time.keys().index(timestamp)
    for docID in training_docs:
        training_docs[docID]['timestamp'] = list(cluster_time.keys()).index(training_docs[docID]['timestamp'])
    return training_docs

def TimestampText(training_docs):
    all_timestamp = list(set([training_docs[docID]['timestamp'] for docID in training_docs]))
    # find timestamp display text: Y/M/D - Y/M/D
    timestamp_text = {}
    for thisTimestamp in all_timestamp:
        allTime = []
        for docID in training_docs:
            timestamp = training_docs[docID]['timestamp']
            if timestamp == thisTimestamp and "time" in training_docs[docID]:
                Y = training_docs[docID]['time'][0]
                if type(Y) != int:
                    Y = 0
                M = training_docs[docID]['time'][1]
                if type(M) != int:
                    M = 0
                D = training_docs[docID]['time'][2]
                if type(D) != int:
                    D = 0
                allTime.append([Y, M, D])

        allTime = sorted(allTime, key=lambda x: x[0]*365+x[1]*30+x[2])
        minTime = allTime[0]
        maxTime = allTime[-1]
        if minTime[0] == maxTime[0] == 0:
            minTime[0] = '0000'
            maxTime[0] = '0000'
        if minTime[1] == maxTime[1] == 0:
            minTime[1] = '00'
            maxTime[1] = '00'
        if minTime[2] == maxTime[2] == 0:
            minTime[2] = '00'
            maxTime[2] = '00'
        timestamp_text[thisTimestamp] = f"{minTime[0]}/{minTime[1]}/{minTime[2]} - {maxTime[0]}/{maxTime[1]}/{maxTime[2]}"

    return timestamp_text



def TrainModelOverTime(training_docs, MODEL, timestamp_text):
    timestamps = [training_docs[docID]['timestamp'] for docID in training_docs]
    if len(set(timestamps)) <= 1:
        TOPIC_TIME_RES = None
        st.warning("有且僅有一個時間戳，無法訓練隨時間變化的主題模型 \n\n Only one timestamp found, cannot train model over time.")
        return TOPIC_TIME_RES
    
    docs = [training_docs[docID]['content'] for docID in training_docs]
    Model_df = MODEL.topics_over_time(docs, timestamps)
    #Model_df['Timestamp'] = [f"{timestamp}: {timestamp_text[timestamp].replace(' ', '')}" for timestamp in Model_df['Timestamp']]
    Model_df['Timestamp'] = [f"{timestamp_text[timestamp].replace(' ', '')}" for timestamp in Model_df['Timestamp']]
    
    #st.write(Model_df)
    vis = VisualizeTopicOverTime(Model_df)
    TOPIC_TIME_RES = {'Model_df': Model_df, 'Figs': {'Variation of Topics Over Time': vis}}
    return TOPIC_TIME_RES

def VisualizeTopicOverTime(Model_df):
    timestamp = Model_df['Timestamp'].unique()
    topics = Model_df['Topic'].unique()
    max_freq = Model_df['Frequency'].max()
    Model_df_with_zero = Model_df.copy()
    figs = {}
    i = 0
    for topic in topics:
        i += 1
        topicID = topic
        df = Model_df[Model_df['Topic'] == topicID]
        # add missing timestamp with 0 frequency
        for ts in timestamp:
            if ts not in df['Timestamp'].values:
                df = pd.concat([df, pd.DataFrame([{'Topic': topicID, 'Words': '', 'Frequency': 0, 'Timestamp': ts, 'Name': f"Topic {topicID}"}])], ignore_index=True)
                Model_df_with_zero = pd.concat([Model_df_with_zero, pd.DataFrame([{'Topic': topicID, 'Words': '', 'Frequency': 0, 'Timestamp': ts, 'Name': topic}])], ignore_index=True)
        df = df.sort_values(by='Timestamp')
        figs[topic] = go.Figure(data=[go.Bar(x=df['Timestamp'], y=df['Frequency'], marker_color=st.session_state.topics[f'Topic {topic}']['COLOR'])])
        figs[topic].update_layout(title=f"Topic {topicID} Over Time", xaxis_title="Timestamp", yaxis_title="Frequency")
    
    
    # use an empty key for a plot that contains all as subplots
    Model_df = Model_df_with_zero
    Model_df = Model_df.sort_values(by=['Topic', 'Timestamp'])
    figs[''] = go.Figure()
    num_topics = len(topics)
    rows = num_topics
    cols = 1
    figs[''] = make_subplots(rows=num_topics, cols=1, subplot_titles=[f'Topic{topic}' for topic in topics])
    for i in range(num_topics):
        topic = topics[i]
        topicID = topic
        df = Model_df[Model_df['Topic'] == topicID]
        row = (i // cols) + 1
        col = (i % cols) + 1
        figs[''].add_trace(go.Bar(x=df['Timestamp'], y=df['Frequency'], marker_color=st.session_state.topics[f'Topic {topic}']['COLOR'], name=f'Topic{topic}'), row=row, col=col)
        figs[''].update_yaxes(range=[0, max_freq], row=row, col=col)
    figs[''].update_layout(height=rows * 400, title_text="Topics Over Time")

    return figs

    
def ExportTopicOverTimeData(topic_time, training_docs, timestamp_text):
    # topic model over time
    Model_df = topic_time['Model_df'].copy()
    # original df has colum: "Topic", "Words", "Frequnecy", "Timestamp", "Name"
    # modify to "Topic"(replace by original Name), "Words", "Frequnecy", "Timestamp", "Name", "Timestamp_detail"(use Timestamp_text)
    Model_df['Topic'] = [f"{topic}" for topic in Model_df['Topic']]
    # Model_df = Model_df.drop(columns=['Name'])
    
    if timestamp_text:
        #st.write(timestamp_text)
        #for timestamp in Model_df['Timestamp']:
            #st.write(timestamp)
        #Model_df['Timestamp_detail'] = [timestamp_text[timestamp] for timestamp in Model_df['Timestamp']]
        csv_model = Model_df.to_csv(index=False)
        # document time info
        Doc_df = pd.DataFrame.from_dict(training_docs, orient='index').reset_index().rename(columns={'index': 'doc_id'})
        Doc_df = Doc_df.drop(columns=['content'])
        Doc_df['time'] = Doc_df['time'].apply(lambda x: f"{x[0]}-{x[1]}-{x[2]}")
        csv_doc = Doc_df.to_csv(index=False)
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            zip_file.writestr(f'topic_over_time.csv', csv_model)
            zip_file.writestr(f'document__time data.csv', csv_doc)

        st.download_button(
        label="Download all data as ZIP",
        data=zip_buffer.getvalue(),
        file_name=f'topic_over_time_data.zip',
        mime='application/zip',
        )

    else:
        csv_model = Model_df.to_csv(index=False)
        st.download_button(
            label="Download data as CSV",
            data=csv_model,
            file_name=f'topic_over_time_.csv',
            mime='text/csv',
        )




def Display_Time(training_docs, TopicOverTime, timestamp_text):
    timestamp_color = {}
    colors = ["#0056d6", "#3571dc", "#5689e5", "#739eee", "#8eb2f6", "#a8c6fe",
          "#669c35", "#7aac50", "#8fbc69", "#a3cb84", "#b8da9c", "#cce8b5",
          "#ff6a00", "#ff8a22", "#ffa440", "#ffbc5f", "#fed083", "#ffe4a8",
          "#ff6251", "#ff7696", "#ff8ac0", "#ff9ddd", "#f8b3f2", "#f1c9fe",
          "#00a3d7", "#00b5c8", "#00c8ac", "#52d66d", "#b3d336", "#fec700"
          ]
    for docID in training_docs:
        timestamp = training_docs[docID]['timestamp']
        if timestamp not in timestamp_color:
            if len(timestamp_color) <= len(colors):
                timestamp_color[timestamp] = colors[len(timestamp_color) % len(colors)]
            else:
                timestamp_color[timestamp] = "#%06x" % random.randint(0, 0xFFFFFF)

    if TopicOverTime != 'Timestamp':
        doc_time = rescale_time(training_docs) 
        scater_plot_data = [] # x = scaled time, color = timestamp, hover = docID: Y-M-D
        for docID in doc_time:
            scater_plot_data.append([doc_time[docID]['scaledTime'], training_docs[docID]['timestamp'], f"{docID}: {training_docs[docID]['time'][0]}-{training_docs[docID]['time'][1]}-{training_docs[docID]['time'][2]}"])

        # Generate scatter plot
        x = [data[0] for data in scater_plot_data]
        colors = [timestamp_color[training_docs[docID]['timestamp']] for docID in doc_time]

        # Visualize
        hovertext = [data[2] for data in scater_plot_data]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=[0]*len(x), mode='markers', marker=dict(color=colors, size=10), text=hovertext, hoverinfo='text'))
        fig.update_layout(title=f'Document Timestamps (K = {len(set([training_docs[docID]["timestamp"] for docID in training_docs]))})')
        fig.update_xaxes(title_text='Scaled Time (days)')
        fig.update_yaxes(visible=False)
        # fig.update_xaxes(visible=False)

        ############################################################
        scater_plot_data = [] # x = scaled time, color = timestamp, hover = docID: Y-M-D
        for docID in doc_time:
            if "time" in training_docs[docID]:
                scater_plot_data.append([doc_time[docID]['scaledTime'], training_docs[docID]['timestamp'], f"{docID}: {training_docs[docID]['time'][0]}-{training_docs[docID]['time'][1]}-{training_docs[docID]['time'][2]}"])

        # Generate scatter plot
        x = [data[0] for data in scater_plot_data]
        colors = [timestamp_color[training_docs[docID]['timestamp']] for docID in doc_time]

        # Visualize
        hovertext = [data[2] for data in scater_plot_data]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=[0]*len(x), mode='markers', marker=dict(color=colors, size=10), text=hovertext, hoverinfo='text', showlegend=False))

        # Add legend
        unique_timestamps = list(set([training_docs[docID]['timestamp'] for docID in training_docs]))
        legend_x = [0.05] * len(unique_timestamps)
        legend_y = list(range(len(unique_timestamps), 0, -1))
        legend_colors = [timestamp_color[timestamp] for timestamp in unique_timestamps]
        legend_text = [f"{timestamp}: {timestamp_text[timestamp]}" for timestamp in unique_timestamps]

        fig.add_trace(go.Scatter(x=legend_x, y=legend_y, mode='markers+text', marker=dict(color=legend_colors, size=10), text=legend_text, textposition='middle right', showlegend=False))

        fig.update_layout(title=f'Document Timestamps (K = {len(unique_timestamps)})')
        fig.update_xaxes(title_text='Scaled Time (days)')
        fig.update_yaxes(visible=False)
        # fig.update_xaxes(visible=False)

        # Adjust layout to make space for the legend
        fig.update_layout(margin=dict(r=150))


    else: # no figure
        fig = None
    return fig
