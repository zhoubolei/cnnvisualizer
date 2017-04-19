# sample code to use the synthetic receptive field of unit to segment images.

Synthetic receptive field is generated using the average actual size of receptive field from the results in our ICLR'15 paper. We found that the segmentation result is comparable to the segmentation using the actual size of RF, which is computationally expensive to estimate. For quick visualization, you could assume each unit has the same size of receptive field, and use the synthetic receptive field to segment the image.

