import glob
import numpy as np
txt_files = glob.glob("testruns/*.txt")

# print(txt_files)
def parse(file):
    lines = []
    control = []
    transfer_no_t = []
    transfer = []
    with open(file) as f:
        line = f.readline()
        while line:
            #print(line)
            reverse = line[::-1]
            index = reverse.find(':') -1
            x = reverse[1:index]
            control.append(int(x[::-1]))
            line = f.readline()

    print(file, ':')
    # print("Array of Steps: ", control)
    control = control[:50]
    print("Average: ", np.average(control))
    print("St. Dev: ", np.std(control))
    print("Median: ", np.median(control))
    n = 5
    print("Min 10: ", np.mean(sorted(control)[:n]))
    print("Max 10: ", np.mean(sorted(control)[-n:]))


for file in txt_files:
    parse(file)

    # print('noT:')
    # print("Array of Steps: ", transfer_no_t)
    # print("Average: ", np.average(transfer_no_t))
    # print("St. Dev: ", np.std(transfer_no_t))
    # print("Median: ", np.median(transfer_no_t))
    #
    # print('T:')
    # print("Array of Steps: ", transfer)
    # print("Average: ", np.average(transfer))
    # print("St. Dev: ", np.std(transfer))
    # print("Median: ", np.median(transfer))
