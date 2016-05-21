import getopt
import numpy as np
import sys

def generate_indices(mode, upper, lower, exclude=[], seed=None):
    if mode == "random":
        #NOTE: If data set was so large that we couldn't keep total list of indices in memory,
        # then we would simply draw with replacement.
        #Generate a random set of indices
        idx_array = np.arange(lower, upper)
        random_state = np.random.RandomState(seed)
        random_state.shuffle(idx_array)
        for index in idx_array:
            print index # Finally pass on to the stream!  
        #print "STOP"
    elif mode == "xvalidation":    
	idx_array = np.arange(lower, upper)
        #shuffle by same seed as the many times before
        random_state = np.random.RandomState(seed)
        random_state.shuffle(idx_array)
	for interval in exclude:
            del idx_array[interval[0]:interval[1]]            
        #Now do another random shuffle (note: use different random state!) on the unexcluded data
        np.random.shuffle(idx_array)
        for index in idx_array:
            print index # Finally pass on the stream!  
        #print "STOP"
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
                 -m mode
                 -u range upper
                 -l range lower
              """
        pass

    try:
        opts, args = getopt.getopt(argv[1:], "hm:u:l:s:e:", ["help"])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    mode = None
    upper = None
    lower = None
    seed = None
    exclude = None
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-m"):
            mode = arg
        elif opt in ("-u"):
            upper = arg
        elif opt in ("-l"):
            lower = arg
        elif opt in ("-s"):
            seed = arg
        elif opt in ("-e"):
            exclude = arg

    assert mode in ["random", "xvalidation"], "Invalid mode "+arg
    try:
        upper = int(upper)
        lower = int(lower)
        if seed is not None:
            seed = int(seed)
    except ValueError:
        print "Upper and Lower, Seed need to be integers!"
        sys.exit(2)
    assert upper >= 0 and lower >= 0 and upper > lower, "Upper needs to be larger than lower and both need to be non-zero!"
    if seed is not None:
        assert seed >= 0, "Seed needs to be non-negative!"
    if exclude is not None:
        pass
        #exclude=
        #assert isinstance(exclude, list)
        #for interval in exclude:
        #    assert isinstance(interval, tuple), "error: tuple " + str()

    #finally generate indices
    generate_indices(mode, upper, lower, seed=seed)

if __name__ == "__main__":
    main(sys.argv)


