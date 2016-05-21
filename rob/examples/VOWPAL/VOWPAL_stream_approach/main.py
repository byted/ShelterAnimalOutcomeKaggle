from bisect import bisect
import getopt
import numpy as np
import subprocess
import sys
import zipfile

def Q1c():
     """
     Q1c: Train VW on the training data. Retrieve average loss.
     """
     archive_filename = "train.csv.zip"
     training_filename = "train.csv" # This file is at the root of archive_filename
     nr_header_lines = 1 # one header line to skip
     nr_example_rows = -1 # to be set later
 
     #have to count number of examples the hard way (if uncompressed, seek to file end and get last line id from Kaggle file)!
     with zipfile.ZipFile(archive_filename) as z:
         with z.open(training_filename, 'r') as f:                  
             #Calculate number of lines in file 
             #Unfortunately, as file is compressed, cannot use seek to end of file to read line number of last line directly (which is nicely provided by Kaggle training file)!
             count = 0
             for _ in f:
                count += 1
             nr_example_rows = count - nr_header_lines

     print nr_example_rows

     #traverse training file in random order and pipe results via adapter to VW
     idx_generator_proc = subprocess.Popen(('python', 'idx_generator.py', '-m', 'random', '-u', str(nr_example_rows) , '-l', str(1)), stdout=subprocess.PIPE)
     traversal_proc = subprocess.Popen(('python', 'traverse.py', '-f', 'train.csv.zip___train.csv', '-c', str(500)), stdin=idx_generator_proc.stdout, stdout=subprocess.PIPE)
     adapter_proc = subprocess.Popen(('python', 'adapter.py'), stdout=subprocess.PIPE, stdin=traversal_proc.stdout)
     output_proc = subprocess.check_output(('vw', '--oaa', '9', '--loss_function', 'logistic'), stdin=adapter_proc.stdout)
     idx_generator_proc.wait()
     traversal_proc.wait()
     adapter_proc.wait()
 
def Q1d():
     """
     Q1d: Perform cross-validation
     """
     pass

def main(argv):
    """
    To be called by correctors. Executes code demonstrating answers to specific problem sheet questions which can be specified through command line argument.
    """ 

    # Handle cmd line args
    train_filename = None
    def usage():
        print """The following command line arguments are available:"
                 --help, -h: print usage info
                 -q 1c: run code for question 1c
                 -q 1d: run code for question 1d
              """
        pass

    try:
        opts, args = getopt.getopt(argv[1:], "hq:", ["help"])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-q",):
            if arg == "1c":
                print "Executing code for question 1c..."
                Q1c()
            elif arg == "1d":
                print "Executing code for question 1d..."
                Q1d()
            else:
                print "Question ", arg, " not available!"
                sys.exit(2)


if __name__ == "__main__":
    main(sys.argv)


#adapter = subprocess.Popen(('python', 'adapter.py'), stdout=subprocess.PIPE)
#output = subprocess.check_output(tuple(" ".split("vw --oaa 9 -f data.model --cache_file cache_file, --loss_function logistic --passes 20")), stdin=adapter.stdout)
#ps.wait()

