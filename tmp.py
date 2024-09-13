import json

data = json.load(open("datasets/formalgeo7k/files/theorem_GDL-source.json", 'r', encoding='utf-8'))
theo_list = []
for key in data['Theorems']:
    theo_name = key.split('(')[0]
    theo_list.append(theo_name)
    
with open("doc/theorems.txt", 'w', encoding='utf-8') as f:
    for theo in theo_list:
        f.write(f"{theo}\n")