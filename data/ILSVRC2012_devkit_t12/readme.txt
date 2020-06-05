=================================================
Introduction
=================================================

This is the documentation of the ILSVRC2012 Development Kit.

Please contact ilsvrc2012@image-net.org for questions, comments,
or bug reports.


=================================================
Data
=================================================

There are three types of image data for this competition: training
data from ImageNet (TRAINING), validation data specific to this
competition (VALIDATION), and test data specific to this competition
(TEST).  There is no overlap in the three sources of data: TRAINING,
VALIDATION, and TEST.  All three sets of data contain images of 1000
categories of objects.  The categories correspond 1-1 to a set of 1000
synsets (sets of synonymous nouns) in WordNet.  An image is in a
particular category X, where X is a noun synset, if the image contains
an X. See [1] for more details of the collection and
labeling strategy.

The 1000 synsets are selected such that there is no overlap between
synsets, for any sysnets i and j, i is not an ancestor of j in the
WordNet hierarchy. We call these synsets "low level synsets".

Those 1000 synsets are part of the larger ImageNet hierarchy and we
can consider the subset of ImageNet containing the 1000 low level
synsets and all of their ancestors. There are 860 such ancestor
synsets, which we refer to as "high level synsets". In this hierarchy,
all the low level synsets are "leaf" nodes and the high level synsets
are "internal" nodes.

Note that the low level synsets may have children in ImageNet, but for
ILSVRC 2012 we do not consider their child subcategories. The
hierarchy here can be thought of as a "trimmed" version of the
complete ImageNet hierarchy.

Also note that for this competition, all ground truth labels are low
level synsets and entries must predict labels corresponding to one of
the 1000 low level synsets.  Predicting high level synsets is not
considered. There are no additional training images for high level
synsets.

*** Meta Data

All information on the synsets is in the 'synsets' array in data/meta.mat.
To access it in Matlab, type

  load data/meta.mat;
  synsets

and you will see

   synsets =

   1x1 struct array with fields:
       ILSVRC2012_ID
       WNID
       words
       gloss
       num_children
       children
       wordnet_height
       num_train_images

Each entry in the struct array corresponds to a synset,i, and contains
fields:

'ILSVRC2012_ID' is an integer ID assigned to each synset. All the low
level synsets are assigned to an ID between 1 and 1000. All the high
level synsets have ID higher than 1000. The synsets are sorted by
ILSVRC2012_ID in the 'synsets' array, i.e.  synsets(i).ILSVRC2012_ID
== i. For submission of prediction results, ILSVRC2012_ID is used as
the synset labels.

'WNID' is the WordNet ID of a synset. It is used to uniquely identify
a synset in ImageNet or WordNet.  The tar files for training images
are named using WNID. Also it is used in naming individual training
images.

'num_children' is the number of the children in this trimmed
hierarchy. It is zero for all low level synsets and non-zero for high
level synsets.

'children' is an vector listing the ILSVRC2012_IDs of child synsets.

'wordnet_height' is the length of the longest path to a leaf node in
the FULL ImageNet/WordNet hierarchy (leaf nodes in the FULL ImageNet
hierarchy have wordnet_height zero). 

The ILSVRC2012_ID of the root of the hierarchy is 1001, the synset
"entity".


*** Training images

There is a tar file for each synset, named by its WNID. The image files are named 
as x_y.JPEG, where x is WNID of the synset and y is an integer (not fixed width and not
necessarily consecutive). All images are in JPEG format. 

There are a total of 1281167 images for training. The number of images for each 
synset ranges from 732 to 1300. 


*** Validation images

There are a total of 50,000 validation images. They are named as

      ILSVRC2012_val_00000001.JPEG
      ILSVRC2012_val_00000002.JPEG
      ...
      ILSVRC2012_val_00049999.JPEG
      ILSVRC2012_val_00050000.JPEG

There are 50 validation images for each synset.

The ground truth of the validation images is in 
    data/ILSVRC2012_validation_ground_truth.txt,
where each line contains one ILSVRC2012_ID for one image, in the ascending alphabetical 
order of the image file names.


*** Test images

There are a total of 100,000 test images, which will be released separately at a later
time. The test files are named as

      ILSVRC2012_test_00000001.JPEG
      ILSVRC2012_test_00000002.JPEG
      ...
      ILSVRC2012_test_00099999.JPEG
      ILSVRC2012_test_00100000.JPEG

There are 100 test images for each synset.

Ground truth of the test images will not be released during the competition.


*** Bounding Boxes

All images in validation and test and at least 100 images in the training set
have bounding box annotations, in PASCAL VOC format. It can be parsed
using the PASCAL development toolkit. There's one XML file for each image 
with bounding box annotations. If the image filename is X.JPEG, then the 
bounding box file is named as X.xml. 

For more information on the bounding box annotations, visit:

http://www.image-net.org/download-bboxes

If there are multiple instances within an image, every instance 
is guaranteed to have a bounding box. 

You are free to use bounding boxes in the competition, for both 
the classification task and the localization task. 

===============================================================
Submission and evaluation
===============================================================

**** Submission format:

The 100,000 test images will be in the same format, i.e. from 
ILSVRC2012_test_00000001.JPEG to ILSVRC2012_test_0100000.JPEG. 

For task 1, submission of results on test data will consist of a 
text file with one line per image, in the alphabetical order of the 
image file names, i.e. from ILSVRC2012_test_00000001.JPEG to 
ILSVRC2012_test_0100000.JPEG. Each line contains the predicted 
labels, i.e. the ILSVRC2012_IDs ( an integer between 1 and 1000 ) 
of the predicted categories, sorted by confidence in descending 
order. The number of labels per line can vary but must be no more 
than 5.

For task 2, submission is similar to task 1, but in each line,
each of the predicted labels is followed by the detected location
of that object sorted by confidence in descending order. It looks 
as follows :
<label(1)> <xmin(1)> <ymin(1)> <xmax(1)> <ymax(1)>  <label(2)> <xmin(2)> <ymin(2)> <xmax(2)> <ymax(2)> ....

The number of labels per line can vary, but not more than 5 
( extra labels are ignored ).

Example files on the validation data is 

        ./evaluation/demo.val.pred.txt			for Task 1
		./evaluation/demo.val.pred.det.txt		for Task 2

**** Evaluation routines

The Matlab routines for evaluating the submission for Task 1 are

./evaluation/eval_flat.m  

and for Task 2

./evaluation/eval_localization_flat.m  

To see an example of using the routines, start Matlab
in the 'evaluation' folder and type
       demo_eval;

and you will see the following output:

pred_file =

demo.val.pred.txt


ground_truth_file =

../data/ILSVRC2012_validation_ground_truth.txt

Task 1: # guesses  vs flat error
    1.0000    0.9990
    2.0000    0.9980
    3.0000    0.9972
    4.0000    0.9962
    5.0000    0.9950


pred_localization_file =

demo.val.pred.det.txt


ground_truth_file =

../data/ILSVRC2012_validation_ground_truth.txt

Please enter the path to the Validation bounding box annotations directory: ~/AnnoVal/val

localization_ground_truth_dir =

~/AnnoVal/val

Task 2: # guesses  vs flat error
    1.0000    1.0000
    2.0000    0.9999
    3.0000    0.9998
    4.0000    0.9997
    5.0000    0.9996

In this demo, we take top i ( i=1...5) predictions (and ignore the 
rest) from your result file and plot the error as a function 
of the number of guesses.

Note that only the error with 5 guesses will be used to determine the 
winner of this competition for each criteria.

** The demo.val.pred.txt used here is a synthetic result 

====================================================================
References
====================================================================

[1] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li and L. Fei-Fei, ImageNet: 
A Large-Scale Hierarchical Image Database. IEEE Computer Vision and Pattern 
Recognition (CVPR), 2009. 

