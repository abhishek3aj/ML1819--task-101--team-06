
from random import choices
population = [1,2,3]
prob = [.58, .26, .16]
import csv
large = []
medium = []
small = []
with open('../weatherHistory.csv', "r") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        rNum = choices(population, prob)
        print(rNum)
        if rNum == 1:
            large.append(row)
        if rNum == 2:
            medium.append(row)
        if rNum == 3:
            small.append(row)
lfile = open('../weather_large.csv', 'w')
with lfile:
   writer = csv.writer(lfile)
   writer.writerows(large)

mfile = open('../weather_medium.csv', 'w')
with mfile:
   writer = csv.writer(mfile)
   writer.writerows(medium)

sfile = open('../weather_small.csv', 'w')
with sfile:
   writer = csv.writer(sfile)
   writer.writerows(small)
