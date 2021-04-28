import os
sequences = 0
file_count = 0
outname = "bulk.emg"


print("target directory:")
directory = input()


f_list = os.listdir(directory)
if(outname in f_list):
    f_list.remove(outname)

with open(directory + '/' + outname, 'w') as out:
    for f_name in f_list:
        if(".emg" in f_name):
            file_count += 1
            with open(directory + '/' + f_name) as f:
                for line in f:
                    out.write(line)
                    sequences += 1
        else:
            print("incorrect filetype found. Are you sure you selected the right folder?")


print("%i sequences in %i files processed" % (sequences, file_count))
input()