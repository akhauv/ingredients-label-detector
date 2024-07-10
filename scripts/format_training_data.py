'''
A simple script to help format training data for the mobileBert model. It brackets
each line in quotations, adds a comma at the end, and adds an indent, outputting 
in a format useful for data/training_data/training_data.py.
'''

# file paths 
in_path = "./data/training_data/unformatted_data.txt"
out_path = "./data/training_data/formatted_data.txt"

# open files 
f = open(in_path, "r")
w = open(out_path, "w")

# read all lines in in_path and write reformatted lines to out_path
lines = f.readlines()
for line in lines:
    line = line.rstrip()
    if (not line):
        continue

    # reformat backslashes and quotations so lines can be included in code
    line = line.replace('\\\'', '\\\\\'')
    line = line.replace('\'', '\\\'')
    newline = '\t\'' + line + '\',\n'

    # write line
    w.write(newline)

# close files
f.close()
w.close()
