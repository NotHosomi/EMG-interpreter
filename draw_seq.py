import os
import matplotlib.pyplot as plt
import numpy as np

print("Filename:")
#filename = input()
filename = "data.emg"

sample_lengths = []
num_seqs = 0
with open(filename) as f:
    for fseq in f:
        sample_lengths.append(len(fseq.split("!")))
        num_seqs += 1
print("number of sequences: ", num_seqs)

print("Processing...")
data = []
with open(filename) as f:

    fseq = f.readlines()
    for i in range(num_seqs):
        data.append([])
        for j in range(8):
            data[i].append([])

        fsamp = fseq[i].split("!")
        for j in range(sample_lengths[i]):
            fcomp = fsamp[j].split("-")
            if(len(fcomp) < 8):
                continue
            for k in range(3):
                data[i][k].append(float(fcomp[k])/1024.0)
            for k in range(3, 8):
                data[i][k].append(float(fcomp[k]))


print("%i sequences found" % len(data))
#sid = input()

for sid in range(len(data)):
    #print("drawing ", sid)
    length = len(data[sid][0]) * 2
    if(length < 1300):
        length = 1300
    plt.figure(figsize=(length/96, 800/96), dpi=96)
    plt.tight_layout()
    plt.axis([0, len(data[sid][0])-1, 0, 1.01])
    plt.plot(data[sid][0], 'r-')
    plt.plot(data[sid][1], 'g-')
    plt.plot(data[sid][2], 'b-')
    plt.plot(data[sid][3], 'r--')
    plt.plot(data[sid][4], 'g--')
    plt.plot(data[sid][5], 'b--')
    plt.plot(data[sid][6], 'm--')
    plt.plot(data[sid][7], 'c--')
    plt.legend(["Sensor1", "Sensor2", "Sensor3", "Thumb", "Index", "Middle", "Ring", "Little"])
    plt.ylabel("power")
    plt.xlabel("Time")
    plt.savefig("graphs/filename_%i.png" % sid, bbox_inches='tight')
    plt.close()

print("Complete")