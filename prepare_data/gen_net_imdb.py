import numpy as np
import numpy.random as npr
size = 12
net = str(size)
with open('%s/pos_%s.txt'%(net, size), 'r') as f:
    pos = f.readlines()

with open('%s/neg_%s.txt'%(net, size), 'r') as f:
    neg = f.readlines()

with open('%s/part_%s.txt'%(net, size), 'r') as f:
    part = f.readlines()
    
def view_bar(num, total):
    rate = float(num) / total
    rate_num = int(rate * 100)+1
    r = '\r[%s%s]%d%%' % ("#"*rate_num, " "*(100-rate_num), rate_num, )
    sys.stdout.write(r)
    sys.stdout.flush()
    
import sys
import cv2
import os
import numpy as np

cls_list = []
print('\n'+'positive-48')
cur_ = 0
sum_ = len(pos)
for line in pos:
    view_bar(cur_,sum_)
    cur_ += 1
    words = line.split()
    image_file_name = '../48net/'+words[0]+'.jpg'
    im = cv2.imread(image_file_name)
    h,w,ch = im.shape
    if h!=48 or w!=48:
        im = cv2.resize(im,(48,48))
    im = np.swapaxes(im, 0, 2)
    im = (im - 127.5)/127.5
    label    = 1
    roi      = [-1,-1,-1,-1]
    pts	     = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    cls_list.append([im,label,roi])
print('\n'+'negative-48')
cur_ = 0
neg_keep = npr.choice(len(neg), size=600000, replace=False)
sum_ = len(neg_keep)
for i in neg_keep:
    line = neg[i]
    view_bar(cur_,sum_)
    cur_ += 1
    words = line.split()
    image_file_name = '../48net/'+words[0]+'.jpg'
    im = cv2.imread(image_file_name)
    h,w,ch = im.shape
    if h!=48 or w!=48:
        im = cv2.resize(im,(48,48))
    im = np.swapaxes(im, 0, 2)
    im = (im - 127.5)/127.5
    label    = 0
    roi      = [-1,-1,-1,-1]
    pts	     = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    cls_list.append([im,label,roi]) 
import cPickle as pickle
fid = open("../48net/48/cls.imdb",'w')
pickle.dump(cls_list, fid)
fid.close()

roi_list = []
print('\n'+'part-48')
cur_ = 0
part_keep = npr.choice(len(part), size=300000, replace=False)
sum_ = len(part_keep)
for i in part_keep:
    line = part[i]
    view_bar(cur_,sum_)
    cur_ += 1
    words = line.split()
    image_file_name = '../48net/'+words[0]+'.jpg'
    im = cv2.imread(image_file_name)
    h,w,ch = im.shape
    if h!=48 or w!=48:
        im = cv2.resize(im,(48,48))
    im = np.swapaxes(im, 0, 2)
    im -= 128
    label    = -1
    roi      = [float(words[2]),float(words[3]),float(words[4]),float(words[5])]
    pts	     = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    roi_list.append([im,label,roi])
print('\n'+'positive-48')
cur_ = 0
sum_ = len(pos2)
for line in pos2:
    view_bar(cur_,sum_)
    cur_ += 1
    words = line.split()
    image_file_name = '../48net/'+words[0]+'.jpg'
    im = cv2.imread(image_file_name)
    h,w,ch = im.shape
    if h!=48 or w!=48:
        im = cv2.resize(im,(48,48))
    im = np.swapaxes(im, 0, 2)
    im = (im - 127.5)/127.5
    label    = -1
    roi      = [float(words[2]),float(words[3]),float(words[4]),float(words[5])]
    pts	     = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    roi_list.append([im,label,roi])
import cPickle as pickle
fid = open("../48net/48/roi.imdb",'w')
pickle.dump(roi_list, fid)
fid.close()




def start(net,shuffling=False):

    saveFolder = os.path.join(rootPath, "tmp/data/%s/"%(net))
    #tfrecord name
    if net == 'pnet':
        tfFileName = os.path.join(saveFolder, "all.tfrecord")
        gen_tfrecord(tfFileName, net, 'all', shuffling)
    elif net in ['rnet', 'onet']:
        #for n in ['pos', 'neg', 'part', 'landmark']:
        for n in ['landmark']:
            tfFileName = os.path.join(saveFolder, "%s.tfrecord"%(n))
            gen_tfrecord(tfFileName, net, n, shuffling)
    # Finally, write the labels file:
    print('\nFinished converting the MTCNN dataset!')









def parse_args():
    parser = argparse.ArgumentParser(description='Create hard bbox sample...',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--stage', dest='stage', help='working stage, can be pnet, rnet, onet',
                        default='unknow', type=str)
    parser.add_argument('--gpus', dest='gpus', help='specify gpu to run. eg: --gpus=0,1',
                        default='0', type=str)
    args = parser.parse_args()
    return args




if __name__ == "__main__":

    args = parse_args()
    stage = 'rnet'
    if stage not in ['pnet', 'rnet', 'onet']:
        raise Exception("Please specify stage by --stage=pnet or rnet or onet")
    # set GPU
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    start(stage, True)