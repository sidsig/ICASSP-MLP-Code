
import numpy
import numpy.random as random
import os
import pickle
import sys
import utils as U
import pdb


def read_file(filename):
    """
    Loads a file into a list
    """
    file_list=[l.strip() for l in open(filename,'r').readlines()]
    return file_list

def get_folds(filelist, n_folds):
    n_per_fold = len(filelist) / n_folds
    folds = []
    for i in range(n_folds-1):
        folds.append(filelist[i * n_per_fold: (i + 1) * n_per_fold])
    i = n_folds - 1
    folds.append(filelist[i * n_per_fold:])
    return folds

def generate_mirex_list(train_list, annotations):
    out_list = []
    for song in train_list:
        annot = annotations.get(song,None)
        if annot is None:
            print 'No annotations for song %s' % song
            continue
        assert(type('') == type(annot))
        out_list.append('%s\t%s\n' % (song,annot))

    return out_list
            

def make_file_list(gtzan_path, n_folds=5,):
    """
    Generates lists
    """
    audio_path = os.path.join(gtzan_path,'audio')
    out_path = os.path.join(gtzan_path,'lists')
    files_list = []
    for ext in ['.au', '.mp3', '.wav']:
        files = U.getFiles(audio_path, ext)
        files_list.extend(files)
    random.shuffle(files_list)
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    audio_list_path = os.path.join(out_path, 'audio_files.txt')
    open(audio_list_path,'w').writelines(['%s\n' % f for f in files_list])
    
    annotations = get_annotations(files_list)

    ground_truth_path = os.path.join(out_path, 'ground_truth.txt')
    open(ground_truth_path,'w').writelines(generate_mirex_list(files_list, annotations))
    generate_ground_truth_pickle(ground_truth_path)

    folds = get_folds(files_list, n_folds=n_folds)
    
    ### Single fold for quick experiments
    create_fold(0, 1, folds, annotations, out_path)
    
    for n in range(n_folds):
        create_fold(n, n_folds, folds, annotations, out_path)


def create_fold(n, n_folds, folds, annotations, out_path):
    train_path = os.path.join(out_path, 'train_%i_of_%i.txt' % (n+1, n_folds))
    valid_path = os.path.join(out_path, 'valid_%i_of_%i.txt' % (n+1, n_folds))
    test_path = os.path.join(out_path, 'test_%i_of_%i.txt' % (n+1, n_folds))1
    
    test_list = folds[n]
    train_list = []
    for m in range(len(folds)):
        if m != n:
            train_list.extend(folds[m])
    
    open(train_path,'w').writelines(generate_mirex_list(train_list, annotations))
    open(test_path,'w').writelines(generate_mirex_list(test_list, annotations))
    split_list_file(train_path, train_path, valid_path, ratio=0.8)
    
def split_list_file(input_file, out_file1, out_file2, ratio=0.8):
    input_list = open(input_file,'r').readlines()
    
    n = len(input_list)
    nsplit = int(n *ratio)
    
    list1 = input_list[:nsplit]
    list2 = input_list[nsplit:]
    
    open(out_file1, 'w').writelines(list1)
    open(out_file2, 'w').writelines(list2)


def get_annotation(filename):
    genre = os.path.split(U.parseFile(filename)[0])[-1]
    return genre

def get_annotations(files_list):
    annotations = {}
    for filename in files_list:
        annotations[filename] = get_annotation(filename)

    return annotations

def generate_ground_truth_pickle(gt_file):
    gt_path,_ = os.path.split(gt_file)
    tag_file = os.path.join(gt_path,'tags.txt')
    gt_pickle = os.path.join(gt_path,'ground_truth.pickle')
    
    lines = open(gt_file,'r').readlines()
    
    tag_set = set()
    for line in lines:
        filename,tag = line.strip().split('\t')
        tag_set.add(tag)
    tag_list = sorted(list(tag_set))
    open(tag_file,'w').writelines('\n'.join(tag_list + ['']))
    
    tag_dict = dict([(tag,i) for i,tag in enumerate(tag_list)])        
    n_tags = len(tag_dict)

    mp3_dict = {}
    for line in lines:
        filename,tag = line.strip().split('\t')
        tag_vector = mp3_dict.get(filename,numpy.zeros(n_tags))
        if tag != '':
            tag_vector[tag_dict[tag]] = 1.
        mp3_dict[filename] = tag_vector
    pickle.dump(mp3_dict,open(gt_pickle,'w'))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage: python %s gtzan_path [n_folds=10]' % sys.argv[0]
        sys.exit()
    
    gtzan_path = os.path.abspath(sys.argv[1])
    if len(sys.argv) > 2:
        n_folds = int(sys.argv[2])
    else:
        n_folds = 10
        
    make_file_list(gtzan_path, n_folds)