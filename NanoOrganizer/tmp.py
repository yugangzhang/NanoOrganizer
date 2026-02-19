

def get_pattern_position_in_string( string, pattern):
    p = re.search( pattern, string).span()
    return p
    
     
        
def filter_list( flist,  and_list=[], or_list =[], no_list=[] ):
    '''Y.G. Aug 1, 2019
    filter the flist  
    Parameters
    ----------    
        flist: list of string
        and_list: list of string,  only retrun filename containing all the strings
        or_list: list of string,  only retrun filename containing one of the strings
        no_string: list of string,  only retrun filename not containing the string    
    Returns
    -------
        list, filtered flist
    '''
    tifs = flist
    tifs_ = []
    Nor = len(or_list)
    for tif in tifs:
        flag=1
        if len(and_list):
            for string in and_list:                
                if string not in tif:
                    flag *=0
        if len(or_list):
            c=0
            for string in or_list:            
                if string not in tif:
                    c+=1
            if c==Nor:
                flag *= 0
        if len(no_list):       
            for string in  no_list:
                if string in tif:
                    flag *=0            
        if flag:
            tifs_.append( tif )            
    return np.array( tifs_ )



def ls_dir(inDir, and_list=[], or_list =[], no_list=[] ):
    '''Y.G. Aug 1, 2019
    List all filenames in a filefolder  
    Parameters
    ----------    
        inDir: string, fullpath of the inDir
        and_list: list of string,  only retrun filename containing all the strings
        or_list: list of string,  only retrun filename containing one of the strings
        no_string: list of string,  only retrun filename not containing the string    
    Returns
    -------
        list, filtered filenames
    '''
    #tifs = np.array( [f for f in listdir(inDir) if isfile(join(inDir, f))] )
    import glob
    tifs0 = glob.glob(inDir + '*' )
    tifs = [ t.split('/')[-1] for t in tifs0 ] 
    return filter_list( tifs,  and_list=and_list,or_list=or_list,no_list=no_list)


def get_file_timestamp( full_filename, verbose=False ):
    '''YG Octo, 2020
        Get time stamp of a file
    Input:
      full_filename: string,  the path + filename  of this file
    Output:
      timestamp: float, the epoch time, in second
      dt: datetime class
    
    '''
    
    timestamp = os.path.getmtime( full_filename )
    dt = datetime.fromtimestamp(timestamp)
    if verbose:
        print('The datetime for this file: %s is: %s.'%(full_filename, dt ) )
    return timestamp, dt

    