import os
import matplotlib.pyplot as plt

log_dir = "logs/"

# line colour lookup
lookup = {
    0: 'r-',
    1: 'g-',
    2: 'b-',
    3: 'm-',
    4: 'c-',
    5: 'y-',
}

def draw(filename, subdir) :
    print("Processing...")
    data = []
    output_size = 0
    with open(log_dir + subdir + filename) as f :
        fseq = f.readlines()
        for sid in range(len(fseq)) :
            data.append([])
        
            log_entries = fseq[sid].split()
            output_size = int(log_entries.pop(0))
            for j in range(2 * output_size) :
                data[sid].append([])
        
        
            for j in range(int(len(log_entries) / (2 * output_size))) :
                for k in range(2 * output_size) :
                    data[sid][k].append(float(log_entries[j * 2 * output_size + k]))


    #sid = input()
    legend_names = []
    for i in range(output_size) :
        legend_names.append("Output" + str(i + 1))
        legend_names.append("Target" + str(i + 1))

    for sid in range(len(data)) :
        #print("drawing ", sid)
        length = len(data[sid][0]) * 2
        if (length < 1300) :
            length = 1300
        if (length > 4000) :
            length = 4000
        plt.figure(figsize = (length / 96, 800 / 96), dpi = 96)
        plt.tight_layout()
        plt.axis([0, len(data[sid][0]) - 1, 0, 1.01])
        for i in range(output_size) :
            plt.plot(data[sid][2 * i], lookup[i])
            plt.plot(data[sid][2 * i + 1], lookup[i] + '-')
        plt.legend(legend_names)
        plt.ylabel("Activation")
        plt.xlabel("Timestep")
        #plt.savefig("sequence_log_graphs/" + filename + "_" + str(sid) + ".png", bbox_inches = 'tight')
        plt.savefig("graphs/output_logs/" + subdir + filename + ".png", bbox_inches = 'tight')
        plt.close()


#local_dir = "train_seqs/"
#f_list = os.listdir(log_dir + local_dir)
#for f in f_list :
#    draw(f, local_dir)
local_dir = "test_seqs/"
f_list = os.listdir(log_dir + local_dir)
for f in f_list :
    draw(f, local_dir)


print("Complete")



    #plt.plot(data[sid][0], 'r-')
    #plt.plot(data[sid][1], 'g-')
    #plt.plot(data[sid][2], 'b-')
    #plt.plot(data[sid][3], 'm-')
    #plt.plot(data[sid][4], 'c-')
    #plt.plot(data[sid][5], 'r--')
    #plt.plot(data[sid][6], 'g--')
    #plt.plot(data[sid][7], 'b--')
    #plt.plot(data[sid][8], 'm--')
    #plt.plot(data[sid][9], 'c--')
    #plt.legend(["Thumb", "Index", "Middle", "Ring", "Little", "Thumb Target", "Index Target", "Middle Target", "Ring Target", "Little Target"])