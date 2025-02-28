import random
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import zipfile
import plotly.graph_objects as go
from utils.ConstValue import ALL_Timestamp_COLORS
import io
from utils.helper import GetRandomColor 
from plotly.subplots import make_subplots


def rescale_time(TRAIN_DOCs):
    all_years = [TRAIN_DOCs[doc]['time'][0] if type(TRAIN_DOCs[doc]['time'][0]) == int else 0 for doc in TRAIN_DOCs]
    minYear = min(all_years)
    Doc_time = {}
    for docID in TRAIN_DOCs:
        Doc_time[docID] = {}
        year = TRAIN_DOCs[docID]['time'][0]
        if type(year) != int:
            year = 0
        month = TRAIN_DOCs[docID]['time'][1]
        if type(month) != int:
            month = 0
        day = TRAIN_DOCs[docID]['time'][2]
        if type(day) != int:
            day = 0
        scaledTime = (year - minYear) * 365 + month * 30 + day
        Doc_time[docID]['scaledTime'] = scaledTime
    return Doc_time
    

def GenerateTimestamp(TRAIN_DOCs, cluster_K):
    doc_time = rescale_time(TRAIN_DOCs)

    times = np.array([[doc_time[docID]['scaledTime']] for docID in doc_time])

    kmeans = KMeans(n_clusters=cluster_K, random_state=0).fit(times)
    labels = kmeans.labels_

    for i, docID in enumerate(doc_time):
        TRAIN_DOCs[docID]['timestamp'] = labels[i]

    # reorder the cluster by the mean of time
    cluster_time = {}
    for i in range(cluster_K):
        cluster_time[i] = []
    for docID in doc_time:
        cluster_time[TRAIN_DOCs[docID]['timestamp']].append(doc_time[docID]['scaledTime'])
    for i in range(cluster_K):
        cluster_time[i] = np.mean(cluster_time[i])
    cluster_time = dict(sorted(cluster_time.items(), key=lambda item: item[1]))
    # reorder mapping timestamp -> cludter_time.keys().index(timestamp)
    for docID in TRAIN_DOCs:
        TRAIN_DOCs[docID]['timestamp'] = list(cluster_time.keys()).index(TRAIN_DOCs[docID]['timestamp'])
    return TRAIN_DOCs

def TimestampText(TRAIN_DOCs):
    all_timestamp = list(set([TRAIN_DOCs[docID]['timestamp'] for docID in TRAIN_DOCs]))
    # find timestamp display text: Y/M/D - Y/M/D
    timestamp_text = {}
    for thisTimestamp in all_timestamp:
        allTime = []
        for docID in TRAIN_DOCs:
            timestamp = TRAIN_DOCs[docID]['timestamp']
            if timestamp == thisTimestamp and "time" in TRAIN_DOCs[docID]:
                Y = TRAIN_DOCs[docID]['time'][0]
                if type(Y) != int:
                    Y = 0
                M = TRAIN_DOCs[docID]['time'][1]
                if type(M) != int:
                    M = 0
                D = TRAIN_DOCs[docID]['time'][2]
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


def topic_onverTime_bar_chart(Model_df, RUN):
    # df columns: "Topic", "Words", "Frequency", "Timestamp", "Name"
    # foe each topic, create a bar chart with x = Timestamp, y = Frequnecy, color = Topic
    timestamp = Model_df['Timestamp'].unique()
    max_freq = Model_df['Frequency'].max()
    Model_df_with_zero = Model_df.copy()
    figs = {}
    for topic in RUN['TOPICs'].keys():
        topicID = int(topic.split(" ")[-1])
        df = Model_df[Model_df['Topic'] == topicID]
        # add missing timestamp with 0 frequency
        for ts in timestamp:
            if ts not in df['Timestamp'].values:
                df = pd.concat([df, pd.DataFrame([{'Topic': topicID, 'Words': '', 'Frequency': 0, 'Timestamp': ts, 'Name': f"Topic {topicID}"}])], ignore_index=True)
                Model_df_with_zero = pd.concat([Model_df_with_zero, pd.DataFrame([{'Topic': topicID, 'Words': '', 'Frequency': 0, 'Timestamp': ts, 'Name': topic}])], ignore_index=True)
        df = df.sort_values(by='Timestamp')
        figs[topic] = go.Figure(data=[go.Bar(x=df['Timestamp'], y=df['Frequency'], marker_color=RUN['TOPICs'][topic]['COLOR'])])
        if not RUN['TOPICs'][topic]['LABEL']:
            figs[topic].update_layout(title=f"Topic {topicID} Over Time", xaxis_title="Timestamp", yaxis_title="Frequency")
        else:
            figs[topic].update_layout(title=f"Topic {topicID} | {RUN['TOPICs'][topic]['LABEL']} Over Time", xaxis_title="Timestamp", yaxis_title="Frequency")
    
    
    # use an empty key for a plot that contains all as subplots
    Model_df = Model_df_with_zero
    Model_df = Model_df.sort_values(by=['Topic', 'Timestamp'])
    figs[''] = go.Figure()
    num_topics = len(RUN['TOPICs'])
    figs[''] = make_subplots(rows=num_topics, cols=1, subplot_titles=[topic if not RUN['TOPICs'][topic]['LABEL'] else f"{topic} | {RUN['TOPICs'][topic]['LABEL']}" for topic in RUN['TOPICs']])
    for i, topic in enumerate(RUN['TOPICs'].keys()):
        topicID = int(topic.split(" ")[-1])
        df = Model_df[Model_df['Topic'] == topicID]
        row = i + 1
        col = 1
        figs[''].add_trace(go.Bar(x=df['Timestamp'], y=df['Frequency'], marker_color=RUN['TOPICs'][topic]['COLOR'], name=topic), row=row, col=col)
        figs[''].update_yaxes(range=[0, max_freq], row=row, col=col)
    figs[''].update_layout(height=num_topics * 400, title_text="Topics Over Time")

    return figs
        

def TrainModelOverTime(TRAIN_DOCs, MODEL, timestamp_text, RUN): # RUN for colors
    timestamps = [TRAIN_DOCs[docID]['timestamp'] for docID in TRAIN_DOCs]
    if len(set(timestamps)) <= 1:
        TOPIC_TIME_RES = None
        st.warning("Only one timestamp found, cannot train model over time.")
        return TOPIC_TIME_RES
    
    docs = [TRAIN_DOCs[docID]['content'] for docID in TRAIN_DOCs]
    Model_df = MODEL.topics_over_time(docs, timestamps)
    try:
        temp_ts = list(Model_df['Timestamp']).copy()
        dig = len(str(max(temp_ts)))
        Model_df['Timestamp'] = [f"{str(timestamp).zfill(dig)}: {timestamp_text[timestamp].replace(" ", "")}" for timestamp in Model_df['Timestamp']]
    except:
        pass
    vis_line = MODEL.visualize_topics_over_time(Model_df, custom_labels=True)
    vis_all_dict = topic_onverTime_bar_chart(Model_df, RUN)
    TOPIC_TIME_RES = {'Model_df': Model_df, 
                      'Figs': {
                        'Summary Line chart': vis_line,
                        'Individual Bar Chart': vis_all_dict # {CUR_TOPIC: fig} "": all topics
                          } 
                      } 
    return TOPIC_TIME_RES

    
def ExportTopicOverTimeData(RUN, TRAIN_DOCs, CUR_RUN, timestamp_text):
    # topic model over time
    Model_df = RUN['TOPIC_TIME_RES']['Model_df'].copy()
    # original df has colum: "Topic", "Words", "Frequnecy", "Timestamp", "Name"
    # modify to "Topic"(replace by original Name), "Words", "Frequnecy", "Timestamp", "Name", "Timestamp_detail"(use Timestamp_text)
    Model_df['Topic'] = [topic.split(" | ")[0] for topic in Model_df['Name']]
    Model_df = Model_df.drop(columns=['Name'])
    Model_df['Timestamp'] = [int(timestamp.split(': ')[0]) for timestamp in Model_df['Timestamp']]
    
    if timestamp_text:
        Model_df['Timestamp_detail'] = [timestamp_text[timestamp] for timestamp in Model_df['Timestamp']]
        csv_model = Model_df.to_csv(index=False)
        # document time info
        Doc_df = pd.DataFrame.from_dict(TRAIN_DOCs, orient='index').reset_index().rename(columns={'index': 'doc_id'})
        Doc_df = Doc_df.drop(columns=['content'])
        Doc_df['time'] = Doc_df['time'].apply(lambda x: f"{x[0]}-{x[1]}-{x[2]}")
        csv_doc = Doc_df.to_csv(index=False)
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            zip_file.writestr(f'topic_over_time_{CUR_RUN}.csv', csv_model)
            zip_file.writestr(f'document__time data_{CUR_RUN}.csv', csv_doc)

        st.download_button(
        label=f"Download DataOverTime_{CUR_RUN}.zip",
        data=zip_buffer.getvalue(),
        file_name=f'DataOverTime_{CUR_RUN}.zip',
        mime='application/zip',
        )

    else:
        csv_model = Model_df.to_csv(index=False)
        st.download_button(
            label=f"Download DataOverTime_{CUR_RUN}.csv",
            data=csv_model,
            file_name=f'DataOverTime_{CUR_RUN}.csv',
            mime='text/csv',
        )


def Display_Timestamp(TRAIN_DOCs, TopicOverTime, timestamp_text):
    timestamp_color = {}
    for docID in TRAIN_DOCs:
        timestamp = TRAIN_DOCs[docID]['timestamp']
        if timestamp not in timestamp_color:
            if len(timestamp_color) <= len(ALL_Timestamp_COLORS):
                timestamp_color[timestamp] = ALL_Timestamp_COLORS[len(timestamp_color) % len(ALL_Timestamp_COLORS)]
            else:
                timestamp_color[timestamp] = GetRandomColor(ALL_Timestamp_COLORS, ALL_Timestamp_COLORS)

    if TopicOverTime != 'Timestamp':
        doc_time = rescale_time(TRAIN_DOCs) 
        scater_plot_data = [] # x = scaled time, color = timestamp, hover = docID: Y-M-D
        for docID in doc_time:
            scater_plot_data.append([doc_time[docID]['scaledTime'], TRAIN_DOCs[docID]['timestamp'], f"{docID}: {TRAIN_DOCs[docID]['time'][0]}-{TRAIN_DOCs[docID]['time'][1]}-{TRAIN_DOCs[docID]['time'][2]}"])

        # Generate scatter plot
        x = [data[0] for data in scater_plot_data]
        colors = [timestamp_color[TRAIN_DOCs[docID]['timestamp']] for docID in doc_time]

        # Visualize
        hovertext = [data[2] for data in scater_plot_data]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=[0]*len(x), mode='markers', marker=dict(color=colors, size=10), text=hovertext, hoverinfo='text'))
        fig.update_layout(title=f'Document Timestamps (K = {len(set([TRAIN_DOCs[docID]["timestamp"] for docID in TRAIN_DOCs]))})')
        fig.update_xaxes(title_text='Scaled Time (days)')
        fig.update_yaxes(visible=False)
        # fig.update_xaxes(visible=False)

        ############################################################
        scater_plot_data = [] # x = scaled time, color = timestamp, hover = docID: Y-M-D
        for docID in doc_time:
            if "time" in TRAIN_DOCs[docID]:
                scater_plot_data.append([doc_time[docID]['scaledTime'], TRAIN_DOCs[docID]['timestamp'], f"{docID}: {TRAIN_DOCs[docID]['time'][0]}-{TRAIN_DOCs[docID]['time'][1]}-{TRAIN_DOCs[docID]['time'][2]}"])

        # Generate scatter plot
        x = [data[0] for data in scater_plot_data]
        colors = [timestamp_color[TRAIN_DOCs[docID]['timestamp']] for docID in doc_time]

        # Visualize
        hovertext = [data[2] for data in scater_plot_data]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=[0]*len(x), mode='markers', marker=dict(color=colors, size=10), text=hovertext, hoverinfo='text', showlegend=False))

        # Add legend
        unique_timestamps = list(set([TRAIN_DOCs[docID]['timestamp'] for docID in TRAIN_DOCs]))
        legend_x = [0.05] * len(unique_timestamps)
        legend_y = list(range(len(unique_timestamps), 0, -1))
        legend_colors = [timestamp_color[timestamp] for timestamp in unique_timestamps]
        dig = len(str(max(unique_timestamps)))
        legend_text = [f"{str(timestamp).zfill(dig)}: {timestamp_text[timestamp]}" for timestamp in unique_timestamps]

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