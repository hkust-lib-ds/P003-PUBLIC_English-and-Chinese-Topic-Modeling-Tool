import streamlit as st
import pandas as pd


def DisplayTopic(RUN, CUR_TOPIC):
    if not CUR_TOPIC: # display all topics
        topics = RUN['TOPICs'].keys()
    else: # display only the selected topic
        topics = [CUR_TOPIC]

    for topic in topics:
        label = RUN['TOPICs'][topic]['LABEL']
        words = RUN['TOPICs'][topic]['WORDs'].keys()
        color = RUN['TOPICs'][topic]['COLOR']
        if label:
            st.markdown(f"<div style='background-color: {color}; padding: 10px; border-radius: 5px; height: 8em; overflow: auto;'>"
                        f'<span style="font-weight: bold; font-size: 1.2em;">{topic}: {label}</span><br>'
                        f"{', '.join(words)}</div>", unsafe_allow_html=True)
            st.write("")
        else:
            st.markdown(f"<div style='background-color: {color}; padding: 10px; border-radius: 5px; height: 8em; overflow: auto;'>"
                        f'<span style="font-weight: bold; font-size: 1.2em;">{topic}</span><br>'
                        f"{', '.join(words)}</div>", unsafe_allow_html=True)
            st.write("")
        
    # if len(topics) == 1:
    #     st.write("")
    #     statistics = RUN['TOPICs'][topic]['WORDs']
    #     df = pd.DataFrame(statistics.items(), columns=['Word', 'Score'])
    #     st.table(df)

    

# def showPyLDAvis(CUR_RUN, CUR_TOPIC):
#     pass
