import os
import matplotlib.pyplot as plt
import math

print("Filename:")
#filename = input()
filename = "accuracy"
log_dir = "C:/GamesTech/3rd-year/CCTP/EMG-interpreter/EMG-Interpreter/logs/"
file_address = log_dir + filename + ".txt"

print("Processing...")
highest = 0;
train = []
test = []
with open(file_address) as f:
    points = f.read().split()
    for i in range(int(len(points)/2)):
        train_i_acc = float(points[2*i])
        test_i_acc = float(points[2*i + 1])
        train.append(train_i_acc)
        test.append(test_i_acc)
        if(train_i_acc > highest) :
            highest = train_i_acc
        if(test_i_acc > highest) :
            highest = test_i_acc

# round up to 0.5
if(highest<= 0.1) :
    highest = 0.1
else:
    highest *= 2
    highest = math.ceil(highest)
    highest /= 2
    

length = len(train) * 2
if(length < 1300):
    length = 1300
plt.figure(figsize=(length/96, 800/96), dpi=96)
plt.tight_layout()
plt.axis([0, len(train)-1, 0, highest + .01])
plt.plot(train, 'y-')
plt.plot(test, 'b-')
plt.legend(["Training accuracy", "Test accuracy"])
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.savefig("log_graphs/" + filename + ".png", bbox_inches='tight')
plt.close()

print("Complete")