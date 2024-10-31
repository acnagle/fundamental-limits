import json

with open('compressed_contexts.json', 'r') as file:
    data = json.load(file)

for key in data.keys():
    print(f'{key}: {len(data[key])}')
