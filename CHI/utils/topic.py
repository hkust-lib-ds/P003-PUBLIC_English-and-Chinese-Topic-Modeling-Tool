"""
def processed_topics(topic_list):
    processed_topics = []
    colors = [
    "#8A9B9B", "#7A9B8A", "#A1B3B2", "#8E9497", "#B3B8B6", "#7D8F8D",
    "#B5B8B4", "#A59A8B", "#B8B0A7", "#7A7A7A", "#A79BB2", "#8A9B9E",
    "#C1A59A", "#A89E8D", "#8F7B6C", "#B1A89D", "#7D8F8A", "#B8A8A8",
    "#A7A79C", "#7F8C8A", "#B1A1A1", "#B69C9C", "#7DAF9E", "#8BAFAD",
    "#C6A6B1", "#A8BFB1", "#98B1A8", "#B39EB1", "#A79BCE", "#A4B6B4"
]
    for i in range(len(topic_list)):
        processed_topics.append(topic(i, topic_list[i], color = colors[i]))
    return processed_topics
"""
import random
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

def GetRandomColor(used_colors):
    available_colors = list(set(colors) - set(used_colors))
    if available_colors:
        color = random.choice(available_colors)
    else:
        color = None
        for _ in range(100):  # Try up to 100 times to find a unique color
            potential_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
            if potential_color not in used_colors:
                color = potential_color
                break
        if color is None:  # If no unique color found after 100 tries, pick any random color
            color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
    return color

def processed_topics(topic_list, model):
    topics = {}
    default_label = []
    used_colors = []
    for key in topic_list.keys():
        default_label.append(f'Topic {key}')
        topics[f'Topic {key}'] = { 'LABEL': None, 'COLOR': GetRandomColor(used_colors), 'WORDs': {} }
        words = topic_list[key] 
        topics[f'Topic {key}']['WORDs'] = {word: weight for word, weight in words}
        used_colors.append(topics[f'Topic {key}']['COLOR'])
    model.set_topic_labels(default_label)
    return topics

'''
class topic():
    def __init__(self, topic_id, words_per_topic, label = None, color = None):
        """
        :param words: list of words belonging to this topic
        :param label: customized label for this topic
        :param color: color to represent this topic in visualization
        """
        self.topic_id = topic_id
        self.words_per_topic = words_per_topic
        self.label = label
        self.color = color


def updatedLabels(topic_list, topic_id, topic_label):
    for topic in topic_list:
        if topic.topic_id == topic_id:
            topic.label = topic_label
'''      
