import os

path1 = "/Users/pablo/Desktop/nl2-project/unet/cropped/CBIS-MASS-Cropped-FINAL-IMG/"
path2 = "/Users/pablo/Desktop/nl2-project/unet/cropped/CBIS-MASS-Cropped-FINAL-MSK/"

entries1 = os.listdir(path1)
entries1.sort()

entries2 = os.listdir(path2)
entries2.sort()

print(len(entries1))
print(len(entries2))

found = False

ids = []

for e1 in entries2:
    s = e1.split('CROPPED-')[1]
    ids.append(s)

finalims = "/Users/pablo/Desktop/nl2-project/CBIS/CBIS-MASS-SELECTED-IMGS/"
finalms = "/Users/pablo/Desktop/nl2-project/CBIS/CBIS-MASS-SELECTED-MASKS/"

entries3 = os.listdir(finalims)
entries3.sort()

entries4 = os.listdir(finalms)
entries4.sort()

print(len(entries3))
print(len(entries4))

f = []
for im in entries4:
    if im not in ids:
        os.remove(finalms + im)

