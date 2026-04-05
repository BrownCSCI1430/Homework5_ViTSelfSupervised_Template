# 15-Scenes (CSCI 1430 Split)

This is a balanced resplit of the 15-scenes dataset (Lazebnik et al., 2006) for use in CSCI 1430 at Brown University.

## Split

| Split | Images per class | Total |
|-------|-----------------|-------|
| train | 100 | 1,500 |
| val   |  50 |   750 |
| test  |  50 |   750 |

## Why not the standard split?

The standard Lazebnik split uses 100 images per class for training and all remaining images for testing. This creates two problems for a homework setting:

1. **No validation set.** Students need a validation set to make architecture and hyperparameter decisions without touching test data. Without one, they inevitably tune on test — learning the wrong lesson.

2. **Unbalanced test classes.** The standard test set ranges from 110 to 310 images per class, which means some classes are weighted 3x more than others in the overall accuracy metric.

This resplit pools all images, shuffles with a fixed seed (1430), and draws 100/50/50 per class. Every class contributes equally to every metric. Unused images (the remainder beyond 200 per class) are dropped.

## Source

Lazebnik, S., Schmid, C., & Ponce, J. (2006). Beyond Bags of Features: Spatial Pyramid Matching for Recognizing Natural Scene Categories. *CVPR 2006*.