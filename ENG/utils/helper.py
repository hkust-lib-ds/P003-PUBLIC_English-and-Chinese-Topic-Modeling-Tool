import random
from utils.ConstValue import ALL_COLORS

def BoldDoc(content, words):
    splited_content = content.split(' ')
    for word in words:
        if word in content.split(' '):
            splited_content = [f'<span style="font-weight: bold; font-size: 1.2em;">{word}</span>' if x == word else x for x in splited_content]
    content = " ".join(splited_content)
    return content

def GetRandomColor(used_colors, ALL_COLORS=ALL_COLORS):
    available_colors = list(set(ALL_COLORS) - set(used_colors))
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
