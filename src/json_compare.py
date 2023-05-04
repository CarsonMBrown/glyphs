import json

with open(r"C:\Users\Carson Brown\git\glyphs\dataset\json1.json") as f1:
    json1 = json.loads(f1.read())

with open(r"C:\Users\Carson Brown\git\glyphs\dataset\json2.json") as f2:
    json2 = json.loads(f2.read())

json1.sort(key=lambda x: x["id"])
json2.sort(key=lambda x: x["id"])

for i, (element1, element2) in enumerate(zip(json1, json2)):
    if element1 != element2:
        print(element1)
