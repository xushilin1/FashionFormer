# FashionFormer Output Explanation

The result data structure is organized as a tuple containing three lists.

## First Item: Garment Detection Information
Data Type: List of Arrays  
Length: 46  
Array Dimensionality: (X, 5)  
X: Number of times the particular garment was detected within the image

Each row in an array represents a detected instance of the garment.
The first 4 numbers in each row are always 0 (potentially reserved for bounding box coordinates?).
The last number in each row indicates the certainty level of the garment detection.
If the garment was not detected (probability below a certain threshold), the corresponding array's dimensionality is (0, 5).

```
result[0][4]
array([[0.        , 0.        , 0.        , 0.        , 0.76458955],
       [0.        , 0.        , 0.        , 0.        , 0.12469797],
       [0.        , 0.        , 0.        , 0.        , 0.10306859]],
      dtype=float32)
```

## Second Item: Pixel Mask for Garment Detection
Data Type: List of Boolean Arrays  
Length: 46  
Array Dimensionality: same dimensionality as input image  
Array Length: X (number of detected instances of garments in the image)  

Each Boolean array has the same dimensionality as the input image.
True values correspond to pixels where the corresponding garment was detected.

```
result[1][4][0]
# Note how this image corresponds to result[0][4][0]
```
![Alt text](figs/sample_mask_1.png?raw=true)  

```
result[1][4][1]
# Note how this Image corresponds to result[0][4][1]
```
![Alt text](figs/sample_mask_2.png?raw=true)  

```
result[1][4][2]
# Note how this Image corresponds to result[0][4][2]
```
![Alt text](figs/sample_mask_3.png?raw=true)  


## Third Item: Attribute Detection Information
Data Type: List of Arrays  
Length: 46  
Array Dimensionality: (294,)  

Each array contains probabilities of 294 fashion attributes being present in the detected garment.
Each of the 294 numbers represents the probability of a specific fashion attribute being present in the garment.

