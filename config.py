img_size = (256, 256)

categories = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'trafficlight', 'firehydrant', 'stopsign', 'parkingmeter', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']

batch_size = 16
device = 'cpu'
dmodel = 256
nhead = 8
num_enc_layer = 24
num_dec_layer = 8
num_query = 128

img_root_dir = '/Users/zx/Documents/cv/dataset/VOC2012/JPEGImages'
xml_root_dir = '/Users/zx/Documents/cv/dataset/VOC2012/Annotations'
filelist_files = ['/Users/zx/Documents/cv/dataset/VOC2012/ImageSets/Main/train2017.txt']

epoch = 100