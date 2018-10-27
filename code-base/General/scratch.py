from random import choices
population = [1,2,3]
prob = [.58, .26, .16]
import csv
large = []
medium = []
small = []
with open('/home/abhishek/Downloads/New_Weather.csv', "r") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    header = next(reader, None)
    large.append(header)
    medium.append(header)
    small.append(header)
    for row in reader:
        rNum = choices(population, prob)
        if rNum[0] == 1:
            large.append(row)
        if rNum[0] == 2:
            medium.append(row)
        if rNum[0] == 3:
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
