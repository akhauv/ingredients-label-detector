in_path = "./data/training_data/data_to_format.txt"
out_path = "./data/training_data/formatted_data.txt"

f = open(in_path, "r")
w = open(out_path, "w")

lines = f.readlines()

for line in lines:
    line = line.rstrip()
    if (not line) or (line == "-----------------"):
        continue
    line = line.replace('\\\'', '\\\\\'')
    line = line.replace('\'', '\\\'')
    newline = '\t\t\'' + line + '\',\n'
    w.write(newline)

f.close()
w.close()
