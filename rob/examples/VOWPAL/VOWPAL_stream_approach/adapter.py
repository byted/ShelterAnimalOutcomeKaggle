import fileinput


def main(argv):
    for line in fileinput.input():
        print adapter(line)
        pass

def adapter(kaggle_row):
    split_row = kaggle_row.split(',')
    return b" ".join(filter(None, [split_row[-1].split("_")[-1].strip()+ " ex"+str(split_row[0])+"|f"] + [(str(i)+":"+split_row[i] if split_row[i]!="0" else "") for i in range(1,len(split_row)-1)]))

if __name__ == "__main__":
    main(None)

