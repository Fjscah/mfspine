import os
import os.path as path
import platform

from pip import main
 
sys = platform.system()

def split_filename(filename):
    """
    Args:
        filename (str): filepath

    Returns:
        list: 
    for example 
    f = 'C:\\X\\Data\\foo.txt'
    shortname="foo"
    basename="foo.txt"
    suffix=".txt"
    dirpath="C:\\X\\Data"
    
    f = 'C:\\X\\Data\\foo'
    shortname="foo"
    basename="foo"
    suffix=""
    dirpath="C:\\X\\Data\\foo"
    
    f = 'C:\\X\\Data\\foo\\'
    shortname="foo"
    basename="foo"
    suffix=""
    dirpath="C:\\X\\Data\\foo\\"
    """
    filename=path_to_platform(filename)
    basename=os.path.basename(filename)
    shortname,suffix=os.path.splitext(basename)
    if "." not in suffix:# is dir
        if filename[-1]=="\\" or filename[-1]=="/":
            filename=filename[:-1]
            basename=os.path.basename(filename)
        return filename,basename,suffix
    dirpath=os.path.dirname(filename)
    return dirpath,shortname,suffix

def path_to_platform(filepath):
    """to system filepath format"""
    if sys == "Windows":
        filepath=filepath.replace('/','\\')
        # print("OS is Windows!!!")
    elif sys == "Linux":
        filepath=filepath.replace('\\','/')
        # print("OS is Linux!!!")
    return filepath

def get_parent_dir(path_current="",level=1):    
    '''

    :param path_int: 0表示获取当前路径，1表示当前路径的上一次路径，2表示当前路径的上2次路径，以此类推

    :return: 返回我们需要的绝对路径，是双斜号的绝对路径

    '''

    path_count=level
    if not path_current:
        path_current=os.path.abspath(r".")
    # print('path_current=',path_current)
    path_current_split=path_current.split('\\')
    # print('path_current_split=',path_current_split)
    path_want=path_current_split[0]
   
    for i in range(len(path_current_split)-1-path_count):
        j=i+1
        path_want=path_want+'\\\\'+path_current_split[j]

    return path_want

def create_imgroi_path(imgp,n,note="_roi",folder=""):
    #parentp=get_parent_dir(imgp,2)
    dirp,basep,suffix=split_filename(imgp)
    newp=os.path.join(folder,basep+note+str(n)+suffix)
    return newp

def create_dir(ndir):
    if not os.path.exists(ndir):
        print("create dir : ",ndir)
        os.makedirs(ndir)

def pair_files(list1,list2,suffix=""):
    pairs=[]
    for f in list1:
        d,n,s=split_filename(f)
        for f2 in list2:
            d2,n2,s2=split_filename(f2)
            if (n+suffix)==n2:
                pairs.append([f,f2])
                break
    return pairs
            
def file_list(file_dir,suffix="tif"): 
    files = os.listdir(file_dir) 
    nfiles=[]
    for f in files:
        if f.endswith(suffix):nfiles.append(os.path.join(file_dir,f))
    return nfiles
    
        #print(root) #当前目录路径
        #print(dirs) #当前路径下所有子目录
        #print(files) #当前路径下所有非目录子文件

if __name__=="__main__":
    print(get_parent_dir(path_current=r"myspine\utils\file_base.py",level=1))