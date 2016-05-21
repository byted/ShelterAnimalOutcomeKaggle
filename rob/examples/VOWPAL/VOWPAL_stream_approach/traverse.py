from bisect import bisect
import getopt
import numpy as np
import sys
import zipfile

import myutils

def traverse(filename, chunk_size):
     #check if file is compressed
     archive_filename, training_filename = filename.split("___")
     assert archive_filename is not None and training_filename is not None, "Wrong filename format or uncompressed, in which case still unsupported!"

     with zipfile.ZipFile(archive_filename) as z:
         #Create a shuffled idx array with 0 idx for first example row (need to mind header offset later!)
         #Now pipe the training file in random order through adapter to vowpal wabbit! 
         #As reading lines from file in random fashion is extremely inefficient, we will use a special buffering technique.
         #As file is assumed much larger than memory, and reading order is very erratic, linecache module won't be able to do a lot
         #So will write our own cache-optimized routine! 

         def print_chunk(chunk):
             '''
             Reads a chunk of (randomized) indices from standard input
             Sorts this chunk and uses this to traverse training file sequentially
             Reshuffles chunk to original order
             Prints content of chunk in original order to standard output
             '''
             indexed_chunk = sorted(zip(range(len(chunk)), chunk), key=lambda t: t[1])
             #print indexed_chunk
             row_buffer = [] #will cache all rows belonging to current chunk
             #Now, for each chunk, read through whole training file, then sort the rows back to random order and pipe them via adapter to vw
             with z.open(training_filename, 'r') as f: #have to reopen file in order to reset file pointer as is compressed (else f.seek(0,1))
                for i, line in enumerate(f):
                     #x = bisect(myutils.KeyifyList(indexed_chunk, lambda v: v[1]), i)                   
                     #if x != len(indexed_chunk) and indexed_chunk[x] == i:                     
                     for idx in indexed_chunk:
                         matches = []
                         if idx[1] == i:                           
                             matches.append((idx[0],i))
                         for x,y in matches:   
                             row_buffer.append((x, line))
                     #indices = [i for i, x in enumerate(my_list) if x == "whatever"]
                             
             #bring row_buffer back to random order
             #print sorted(indexed_chunk, key=lambda t: t[0])
             #print row_buffer
             row_buffer = zip(*sorted(row_buffer, key= lambda v:v[0]))[1] #[row_buffer[x[0]] for x in indexed_chunk] #sorted(row_buffer, key=lambda v:indexed_chunk[v][0])
             #now pipe the row buffer to stdout
             for line in row_buffer:
                 print line
         
         #Now receive chunks of indices from standard input, obtain the rows from training file and print the results to standard output!
         chunk_buffer = []
         #for line in sys.stdin: #Receive chunks of indices from standard input
         while True:
             line = sys.stdin.readline()
             #print line, "LINE"
             if not line:                
                return
             #if line == "STOP\n": #Have reached end of stream!
             #   break
             try:
             	chunk_buffer.append(int(line.strip()))
             except ValueError:
                pass #Silently drop garbage index (would be nice to log this incidence at some stage)
             if len(chunk_buffer) == chunk_size:
                print_chunk(chunk_buffer)
                chunk_buffer = [] #clear chunk buffer
         #Don't forget to print any remaining indices in last chunk
         if len(chunk_buffer) > 0:
             print_chunk(chunk_buffer)
     pass     

def main(argv):
    """
    Index generator takes in a range and a mode and outputs a stream of indices to std.out
    """

    # Handle cmd line args
    train_filename = None
    def usage():
        print """The following command line arguments are available:"
                 --help, -h: print usage info
                 -f archivename/filename if compressed, otherwise just filename
                 -n number of header lines to skip
                 -c chunk size in nr of indices (assumed 32bit each)
              """
        pass

    try:
        opts, args = getopt.getopt(argv[1:], "hf:n:c:", ["help"])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    filename = None
    nr_header_lines = 0
    chunk_size = 500 # default chunk size
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-f"):
            filename = arg
        elif opt in ("-n"):
            nr_header_lines = arg
        elif opt in ("-c"):
            chunk_size = arg
    try:
    #   nr_header_lines = int(nr_header_lines)
       chunk_size = int(chunk_size)
    except ValueError:
       print "nr of header lines, chunk size need to be integer!"
       sys.exit(2)
    assert chunk_size > 0, "chunk size needs to be a positive integer!"
    #assert nr_header_lines > 0, "nr of header lines must be positive integer"
    assert filename is not None, "you need to specify the input file name"
    traverse(filename, chunk_size)

if __name__ == "__main__":
    main(sys.argv)


