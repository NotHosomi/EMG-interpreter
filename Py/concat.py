import os
sequences = 0
file_count = 0

outname = "bulk.emg"
in_dir = "C:/GamesTech/3rd-year/CCTP/EMG-interpreter/Training-Data-Builder/data/"
out_dir = "C:/GamesTech/3rd-year/CCTP/EMG-interpreter/EMG-Interpreter/data"

f_list = os.listdir(in_dir)
if(outname in f_list):
    f_list.remove(in_dir)

with open(out_dir + '/' + outname, 'w') as out:
    for f_name in f_list:
        if(".emg" in f_name):
            file_count += 1
            with open(in_dir + '/' + f_name) as f:
                seq = 0
                for line in f:
                    out.write(line)
                    sequences += 1

                    # error detection
                    ErrA = line.find("--")
                    ErrB = line.find("-!")
                    ErrC = line.find("!-")
                    ErrD = line[0] == '-'
                    if(ErrA != -1) :
                        print("Error(A) in sequence %i at %i (%s)" % (seq, ErrA, f_name))
                    if(ErrB != -1) :
                        print("Error(B) in sequence %i at %i (%s)" % (seq, ErrB, f_name))
                    if(ErrC != -1) :
                        print("Error(C) in sequence %i at %i (%s)" % (seq, ErrD, f_name))
                    if(ErrD) :
                        print("Error(D) in sequence %i (%s)" % (seq, f_name))
                    #if(ErrS != -1) :
                    #    print("Error(S) in sequence %i at %i (%s)" % (seq, ErrS, f_name))
                    
                    ## too extensive to fix by hand
                    #if(line.split('!')[0].find('-1') != -1) :
                    #    print("incorrect label at start of sequence %i in file %s" % (seq, f_name))
                    
                    seq += 1
        else:
            print("incorrect filetype found. Are you sure you selected the right folder?")


print("%i sequences in %i files processed" % (sequences, file_count))