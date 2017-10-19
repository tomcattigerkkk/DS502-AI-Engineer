# MXNET-Week1 part2 

Homework for Week1 part2. 

The structures of my network are illustrated in PDFs, **MLP.pdf**, **CNN.pdf**,   
 **CNN_inception.pdf**.

# Some Comparisons 

## Short Result

 	|Comparison              | MLP | CNN | CNN with Inception |
 	|---|---|---|---|
	
	|Acc(5 epoch) with CPU   | .9744 | .9951 | .9932 |

	|CPU Epoch time          | 2464.23 samples/sec | 392.34 samples/sec | 263.65 samples/sec |

	|Acc(5 epoch) with GPU   | .9752 | .9950 | .9939 |
	
	|GPU Epoch Time(optional)| 84755.50 samples/sec | 18796.73 samples/sec | 14494.66 samples/sec |

> GPU: 1 x GTX1060(notebook)

## Details of Test Result

### MLP (CPU)
```textINFO:root:Epoch[5] Batch [100]	Speed: 1176.08 samples/sec	accuracy=0.986535
INFO:root:Epoch[5] Batch [200]	Speed: 2430.13 samples/sec	accuracy=0.986000
INFO:root:Epoch[5] Batch [300]	Speed: 4055.99 samples/sec	accuracy=0.988900
INFO:root:Epoch[5] Batch [400]	Speed: 1949.62 samples/sec	accuracy=0.987300
INFO:root:Epoch[5] Batch [500]	Speed: 2291.39 samples/sec	accuracy=0.988000
INFO:root:Epoch[5] Train-accuracy=0.985455
INFO:root:Epoch[5] Time cost=30.957
INFO:root:Epoch[5] Validation-accuracy=0.973000
INFO:root:Epoch[6] Batch [100]	Speed: 1450.90 samples/sec	accuracy=0.988713
INFO:root:Epoch[6] Batch [200]	Speed: 3146.39 samples/sec	accuracy=0.989700
INFO:root:Epoch[6] Batch [300]	Speed: 3130.45 samples/sec	accuracy=0.991500
INFO:root:Epoch[6] Batch [400]	Speed: 4655.64 samples/sec	accuracy=0.989700
INFO:root:Epoch[6] Batch [500]	Speed: 1238.06 samples/sec	accuracy=0.989500
INFO:root:Epoch[6] Train-accuracy=0.990404
INFO:root:Epoch[6] Time cost=26.439
INFO:root:Epoch[6] Validation-accuracy=0.973600
INFO:root:Epoch[7] Batch [100]	Speed: 1480.58 samples/sec	accuracy=0.989604
INFO:root:Epoch[7] Batch [200]	Speed: 2707.63 samples/sec	accuracy=0.990600
INFO:root:Epoch[7] Batch [300]	Speed: 2947.71 samples/sec	accuracy=0.993800
INFO:root:Epoch[7] Batch [400]	Speed: 3011.25 samples/sec	accuracy=0.992300
INFO:root:Epoch[7] Batch [500]	Speed: 1354.60 samples/sec	accuracy=0.991800
INFO:root:Epoch[7] Train-accuracy=0.992323
INFO:root:Epoch[7] Time cost=28.811
INFO:root:Epoch[7] Validation-accuracy=0.975100
INFO:root:Epoch[8] Batch [100]	Speed: 2455.67 samples/sec	accuracy=0.992970
INFO:root:Epoch[8] Batch [200]	Speed: 1967.27 samples/sec	accuracy=0.993100
INFO:root:Epoch[8] Batch [300]	Speed: 1220.35 samples/sec	accuracy=0.994900
INFO:root:Epoch[8] Batch [400]	Speed: 2097.75 samples/sec	accuracy=0.992500
INFO:root:Epoch[8] Batch [500]	Speed: 3151.68 samples/sec	accuracy=0.993100
INFO:root:Epoch[8] Train-accuracy=0.994545
INFO:root:Epoch[8] Time cost=33.082
INFO:root:Epoch[8] Validation-accuracy=0.975000
INFO:root:Epoch[9] Batch [100]	Speed: 6430.40 samples/sec	accuracy=0.992772
INFO:root:Epoch[9] Batch [200]	Speed: 1369.95 samples/sec	accuracy=0.993900
INFO:root:Epoch[9] Batch [300]	Speed: 1845.73 samples/sec	accuracy=0.994600
INFO:root:Epoch[9] Batch [400]	Speed: 2464.23 samples/sec	accuracy=0.994300
INFO:root:Epoch[9] Batch [500]	Speed: 4136.53 samples/sec	accuracy=0.993100
INFO:root:Epoch[9] Train-accuracy=0.992929
INFO:root:Epoch[9] Time cost=28.218
INFO:root:Epoch[9] Validation-accuracy=0.975500
```

### MLP (GPU)
```text
INFO:root:Epoch[5] Batch [100]	Speed: 85241.07 samples/sec	accuracy=0.986733
INFO:root:Epoch[5] Batch [200]	Speed: 80588.09 samples/sec	accuracy=0.988500
INFO:root:Epoch[5] Batch [300]	Speed: 82805.14 samples/sec	accuracy=0.984800
INFO:root:Epoch[5] Batch [400]	Speed: 88714.27 samples/sec	accuracy=0.988700
INFO:root:Epoch[5] Batch [500]	Speed: 85338.37 samples/sec	accuracy=0.986200
INFO:root:Epoch[5] Train-accuracy=0.988081
INFO:root:Epoch[5] Time cost=0.704
INFO:root:Epoch[5] Validation-accuracy=0.976000
INFO:root:Epoch[6] Batch [100]	Speed: 85349.66 samples/sec	accuracy=0.991386
INFO:root:Epoch[6] Batch [200]	Speed: 83414.79 samples/sec	accuracy=0.988400
INFO:root:Epoch[6] Batch [300]	Speed: 87864.04 samples/sec	accuracy=0.987200
INFO:root:Epoch[6] Batch [400]	Speed: 92283.92 samples/sec	accuracy=0.989900
INFO:root:Epoch[6] Batch [500]	Speed: 89908.32 samples/sec	accuracy=0.989700
INFO:root:Epoch[6] Train-accuracy=0.989495
INFO:root:Epoch[6] Time cost=0.682
INFO:root:Epoch[6] Validation-accuracy=0.974400
INFO:root:Epoch[7] Batch [100]	Speed: 72447.62 samples/sec	accuracy=0.991287
INFO:root:Epoch[7] Batch [200]	Speed: 52577.20 samples/sec	accuracy=0.990300
INFO:root:Epoch[7] Batch [300]	Speed: 78916.70 samples/sec	accuracy=0.989600
INFO:root:Epoch[7] Batch [400]	Speed: 88319.54 samples/sec	accuracy=0.991400
INFO:root:Epoch[7] Batch [500]	Speed: 91110.80 samples/sec	accuracy=0.992400
INFO:root:Epoch[7] Train-accuracy=0.992323
INFO:root:Epoch[7] Time cost=0.793
INFO:root:Epoch[7] Validation-accuracy=0.975800
INFO:root:Epoch[8] Batch [100]	Speed: 86473.90 samples/sec	accuracy=0.992772
INFO:root:Epoch[8] Batch [200]	Speed: 84429.98 samples/sec	accuracy=0.992800
INFO:root:Epoch[8] Batch [300]	Speed: 90288.24 samples/sec	accuracy=0.992300
INFO:root:Epoch[8] Batch [400]	Speed: 84755.50 samples/sec	accuracy=0.992900
INFO:root:Epoch[8] Batch [500]	Speed: 71767.43 samples/sec	accuracy=0.992400
INFO:root:Epoch[8] Train-accuracy=0.991515
INFO:root:Epoch[8] Time cost=0.752
INFO:root:Epoch[8] Validation-accuracy=0.973900
INFO:root:Epoch[9] Batch [100]	Speed: 65143.21 samples/sec	accuracy=0.992079
INFO:root:Epoch[9] Batch [200]	Speed: 65272.16 samples/sec	accuracy=0.992100
INFO:root:Epoch[9] Batch [300]	Speed: 87016.23 samples/sec	accuracy=0.992900
INFO:root:Epoch[9] Batch [400]	Speed: 84412.65 samples/sec	accuracy=0.993600
INFO:root:Epoch[9] Batch [500]	Speed: 80352.19 samples/sec	accuracy=0.993000
INFO:root:Epoch[9] Train-accuracy=0.994343
INFO:root:Epoch[9] Time cost=0.789
INFO:root:Epoch[9] Validation-accuracy=0.975900
```

### CNN (CPU)
```text
INFO:root:Epoch[5] Batch [100]	Speed: 384.29 samples/sec	accuracy=0.999901
INFO:root:Epoch[5] Batch [200]	Speed: 381.35 samples/sec	accuracy=0.999600
INFO:root:Epoch[5] Batch [300]	Speed: 394.82 samples/sec	accuracy=0.999600
INFO:root:Epoch[5] Batch [400]	Speed: 392.34 samples/sec	accuracy=0.999900
INFO:root:Epoch[5] Batch [500]	Speed: 388.45 samples/sec	accuracy=0.999700
INFO:root:Epoch[5] Train-accuracy=0.999697
INFO:root:Epoch[5] Time cost=153.930
INFO:root:Epoch[5] Validation-accuracy=0.994900
INFO:root:Epoch[6] Batch [100]	Speed: 370.53 samples/sec	accuracy=0.999901
INFO:root:Epoch[6] Batch [200]	Speed: 364.70 samples/sec	accuracy=0.999800
INFO:root:Epoch[6] Batch [300]	Speed: 380.60 samples/sec	accuracy=1.000000
INFO:root:Epoch[6] Batch [400]	Speed: 387.10 samples/sec	accuracy=1.000000
INFO:root:Epoch[6] Batch [500]	Speed: 361.01 samples/sec	accuracy=0.999800
INFO:root:Epoch[6] Train-accuracy=0.999798
INFO:root:Epoch[6] Time cost=160.114
INFO:root:Epoch[6] Validation-accuracy=0.995300
INFO:root:Epoch[7] Batch [100]	Speed: 387.00 samples/sec	accuracy=0.999901
INFO:root:Epoch[7] Batch [200]	Speed: 362.60 samples/sec	accuracy=0.999800
INFO:root:Epoch[7] Batch [300]	Speed: 372.92 samples/sec	accuracy=1.000000
INFO:root:Epoch[7] Batch [400]	Speed: 389.56 samples/sec	accuracy=1.000000
INFO:root:Epoch[7] Batch [500]	Speed: 387.93 samples/sec	accuracy=1.000000
INFO:root:Epoch[7] Train-accuracy=0.999899
INFO:root:Epoch[7] Time cost=157.402
INFO:root:Epoch[7] Validation-accuracy=0.995100
INFO:root:Epoch[8] Batch [100]	Speed: 391.88 samples/sec	accuracy=0.999901
INFO:root:Epoch[8] Batch [200]	Speed: 383.27 samples/sec	accuracy=0.999900
INFO:root:Epoch[8] Batch [300]	Speed: 358.27 samples/sec	accuracy=1.000000
INFO:root:Epoch[8] Batch [400]	Speed: 385.75 samples/sec	accuracy=1.000000
INFO:root:Epoch[8] Batch [500]	Speed: 388.52 samples/sec	accuracy=1.000000
INFO:root:Epoch[8] Train-accuracy=0.999899
INFO:root:Epoch[8] Time cost=156.575
INFO:root:Epoch[8] Validation-accuracy=0.995300
INFO:root:Epoch[9] Batch [100]	Speed: 367.07 samples/sec	accuracy=1.000000
INFO:root:Epoch[9] Batch [200]	Speed: 363.71 samples/sec	accuracy=1.000000
INFO:root:Epoch[9] Batch [300]	Speed: 359.56 samples/sec	accuracy=1.000000
INFO:root:Epoch[9] Batch [400]	Speed: 376.85 samples/sec	accuracy=1.000000
INFO:root:Epoch[9] Batch [500]	Speed: 386.04 samples/sec	accuracy=1.000000
INFO:root:Epoch[9] Train-accuracy=0.999899
INFO:root:Epoch[9] Time cost=160.352
INFO:root:Epoch[9] Validation-accuracy=0.995000
```

### CNN (GPU)
```text
INFO:root:Epoch[5] Batch [100]	Speed: 19013.53 samples/sec	accuracy=0.999604
INFO:root:Epoch[5] Batch [200]	Speed: 18914.09 samples/sec	accuracy=0.999700
INFO:root:Epoch[5] Batch [300]	Speed: 19535.04 samples/sec	accuracy=0.999600
INFO:root:Epoch[5] Batch [400]	Speed: 16469.91 samples/sec	accuracy=0.999800
INFO:root:Epoch[5] Batch [500]	Speed: 17501.37 samples/sec	accuracy=0.999800
INFO:root:Epoch[5] Train-accuracy=1.000000
INFO:root:Epoch[5] Time cost=3.265
INFO:root:Epoch[5] Validation-accuracy=0.995000
INFO:root:Epoch[6] Batch [100]	Speed: 19274.11 samples/sec	accuracy=0.999802
INFO:root:Epoch[6] Batch [200]	Speed: 19013.42 samples/sec	accuracy=0.999800
INFO:root:Epoch[6] Batch [300]	Speed: 18800.45 samples/sec	accuracy=0.999800
INFO:root:Epoch[6] Batch [400]	Speed: 19149.69 samples/sec	accuracy=0.999900
INFO:root:Epoch[6] Batch [500]	Speed: 18810.87 samples/sec	accuracy=1.000000
INFO:root:Epoch[6] Train-accuracy=1.000000
INFO:root:Epoch[6] Time cost=3.153
INFO:root:Epoch[6] Validation-accuracy=0.994900
INFO:root:Epoch[7] Batch [100]	Speed: 15976.02 samples/sec	accuracy=0.999901
INFO:root:Epoch[7] Batch [200]	Speed: 19035.95 samples/sec	accuracy=0.999800
INFO:root:Epoch[7] Batch [300]	Speed: 18999.31 samples/sec	accuracy=0.999900
INFO:root:Epoch[7] Batch [400]	Speed: 18781.02 samples/sec	accuracy=0.999900
INFO:root:Epoch[7] Batch [500]	Speed: 19162.92 samples/sec	accuracy=1.000000
INFO:root:Epoch[7] Train-accuracy=1.000000
INFO:root:Epoch[7] Time cost=3.260
INFO:root:Epoch[7] Validation-accuracy=0.994900
INFO:root:Epoch[8] Batch [100]	Speed: 18796.73 samples/sec	accuracy=0.999901
INFO:root:Epoch[8] Batch [200]	Speed: 19091.40 samples/sec	accuracy=0.999700
INFO:root:Epoch[8] Batch [300]	Speed: 19244.37 samples/sec	accuracy=0.999900
INFO:root:Epoch[8] Batch [400]	Speed: 16107.74 samples/sec	accuracy=1.000000
INFO:root:Epoch[8] Batch [500]	Speed: 18769.39 samples/sec	accuracy=1.000000
INFO:root:Epoch[8] Train-accuracy=1.000000
INFO:root:Epoch[8] Time cost=3.258
INFO:root:Epoch[8] Validation-accuracy=0.995000
INFO:root:Epoch[9] Batch [100]	Speed: 19038.10 samples/sec	accuracy=0.999901
INFO:root:Epoch[9] Batch [200]	Speed: 18863.89 samples/sec	accuracy=0.999800
INFO:root:Epoch[9] Batch [300]	Speed: 19402.48 samples/sec	accuracy=1.000000
INFO:root:Epoch[9] Batch [400]	Speed: 18813.74 samples/sec	accuracy=1.000000
INFO:root:Epoch[9] Batch [500]	Speed: 19342.55 samples/sec	accuracy=1.000000
INFO:root:Epoch[9] Train-accuracy=1.000000
INFO:root:Epoch[9] Time cost=3.141
INFO:root:Epoch[9] Validation-accuracy=0.995000
```

### CNN with Inception Layer (CPU)
```text
INFO:root:Epoch[5] Batch [100]	Speed: 225.04 samples/sec	accuracy=0.997525
INFO:root:Epoch[5] Batch [200]	Speed: 212.48 samples/sec	accuracy=0.998600
INFO:root:Epoch[5] Batch [300]	Speed: 216.53 samples/sec	accuracy=0.997800
INFO:root:Epoch[5] Batch [400]	Speed: 221.18 samples/sec	accuracy=0.998700
INFO:root:Epoch[5] Batch [500]	Speed: 222.85 samples/sec	accuracy=0.998900
INFO:root:Epoch[5] Train-accuracy=0.999192
INFO:root:Epoch[5] Time cost=272.176
INFO:root:Epoch[5] Validation-accuracy=0.992800
INFO:root:Epoch[6] Batch [100]	Speed: 219.12 samples/sec	accuracy=0.998020
INFO:root:Epoch[6] Batch [200]	Speed: 218.27 samples/sec	accuracy=0.999500
INFO:root:Epoch[6] Batch [300]	Speed: 211.67 samples/sec	accuracy=0.999600
INFO:root:Epoch[6] Batch [400]	Speed: 214.56 samples/sec	accuracy=0.999000
INFO:root:Epoch[6] Batch [500]	Speed: 213.62 samples/sec	accuracy=0.999500
INFO:root:Epoch[6] Train-accuracy=0.999596
INFO:root:Epoch[6] Time cost=275.526
INFO:root:Epoch[6] Validation-accuracy=0.992300
INFO:root:Epoch[7] Batch [100]	Speed: 274.37 samples/sec	accuracy=0.999208
INFO:root:Epoch[7] Batch [200]	Speed: 258.19 samples/sec	accuracy=0.999000
INFO:root:Epoch[7] Batch [300]	Speed: 264.39 samples/sec	accuracy=0.999700
INFO:root:Epoch[7] Batch [400]	Speed: 273.91 samples/sec	accuracy=0.999600
INFO:root:Epoch[7] Batch [500]	Speed: 261.92 samples/sec	accuracy=0.999600
INFO:root:Epoch[7] Train-accuracy=0.999697
INFO:root:Epoch[7] Time cost=226.740
INFO:root:Epoch[7] Validation-accuracy=0.992900
INFO:root:Epoch[8] Batch [100]	Speed: 277.43 samples/sec	accuracy=0.999802
INFO:root:Epoch[8] Batch [200]	Speed: 261.95 samples/sec	accuracy=1.000000
INFO:root:Epoch[8] Batch [300]	Speed: 259.19 samples/sec	accuracy=1.000000
INFO:root:Epoch[8] Batch [400]	Speed: 254.00 samples/sec	accuracy=0.999700
INFO:root:Epoch[8] Batch [500]	Speed: 267.84 samples/sec	accuracy=0.999600
INFO:root:Epoch[8] Train-accuracy=0.999899
INFO:root:Epoch[8] Time cost=225.970
INFO:root:Epoch[8] Validation-accuracy=0.993800
INFO:root:Epoch[9] Batch [100]	Speed: 263.65 samples/sec	accuracy=1.000000
INFO:root:Epoch[9] Batch [200]	Speed: 280.05 samples/sec	accuracy=1.000000
INFO:root:Epoch[9] Batch [300]	Speed: 266.33 samples/sec	accuracy=1.000000
INFO:root:Epoch[9] Batch [400]	Speed: 277.69 samples/sec	accuracy=0.999800
INFO:root:Epoch[9] Batch [500]	Speed: 275.11 samples/sec	accuracy=1.000000
INFO:root:Epoch[9] Train-accuracy=0.999899
INFO:root:Epoch[9] Time cost=220.556
INFO:root:Epoch[9] Validation-accuracy=0.994200
```

### CNN with Inception Layer (GPU)
```text
INFO:root:Epoch[5] Batch [100]	Speed: 14596.73 samples/sec	accuracy=0.999010
INFO:root:Epoch[5] Batch [200]	Speed: 14419.86 samples/sec	accuracy=0.998700
INFO:root:Epoch[5] Batch [300]	Speed: 13891.50 samples/sec	accuracy=0.999000
INFO:root:Epoch[5] Batch [400]	Speed: 13427.04 samples/sec	accuracy=0.999500
INFO:root:Epoch[5] Batch [500]	Speed: 12513.90 samples/sec	accuracy=0.999600
INFO:root:Epoch[5] Train-accuracy=0.999293
INFO:root:Epoch[5] Time cost=4.351
INFO:root:Epoch[5] Validation-accuracy=0.992800
INFO:root:Epoch[6] Batch [100]	Speed: 12436.48 samples/sec	accuracy=0.999505
INFO:root:Epoch[6] Batch [200]	Speed: 12878.02 samples/sec	accuracy=0.999800
INFO:root:Epoch[6] Batch [300]	Speed: 13157.78 samples/sec	accuracy=0.999500
INFO:root:Epoch[6] Batch [400]	Speed: 14329.99 samples/sec	accuracy=0.999800
INFO:root:Epoch[6] Batch [500]	Speed: 14047.07 samples/sec	accuracy=1.000000
INFO:root:Epoch[6] Train-accuracy=0.999798
INFO:root:Epoch[6] Time cost=4.462
INFO:root:Epoch[6] Validation-accuracy=0.994200
INFO:root:Epoch[7] Batch [100]	Speed: 14204.94 samples/sec	accuracy=0.999802
INFO:root:Epoch[7] Batch [200]	Speed: 14153.76 samples/sec	accuracy=0.999900
INFO:root:Epoch[7] Batch [300]	Speed: 14028.08 samples/sec	accuracy=1.000000
INFO:root:Epoch[7] Batch [400]	Speed: 13875.20 samples/sec	accuracy=0.999900
INFO:root:Epoch[7] Batch [500]	Speed: 14128.55 samples/sec	accuracy=0.999900
INFO:root:Epoch[7] Train-accuracy=0.999899
INFO:root:Epoch[7] Time cost=4.252
INFO:root:Epoch[7] Validation-accuracy=0.994200
INFO:root:Epoch[8] Batch [100]	Speed: 14237.25 samples/sec	accuracy=0.999901
INFO:root:Epoch[8] Batch [200]	Speed: 14189.98 samples/sec	accuracy=0.999800
INFO:root:Epoch[8] Batch [300]	Speed: 14114.18 samples/sec	accuracy=1.000000
INFO:root:Epoch[8] Batch [400]	Speed: 14080.20 samples/sec	accuracy=1.000000
INFO:root:Epoch[8] Batch [500]	Speed: 14183.84 samples/sec	accuracy=0.999900
INFO:root:Epoch[8] Train-accuracy=1.000000
INFO:root:Epoch[8] Time cost=4.231
INFO:root:Epoch[8] Validation-accuracy=0.994100
INFO:root:Epoch[9] Batch [100]	Speed: 14007.70 samples/sec	accuracy=0.999901
INFO:root:Epoch[9] Batch [200]	Speed: 13581.53 samples/sec	accuracy=1.000000
INFO:root:Epoch[9] Batch [300]	Speed: 13892.59 samples/sec	accuracy=1.000000
INFO:root:Epoch[9] Batch [400]	Speed: 14339.62 samples/sec	accuracy=1.000000
INFO:root:Epoch[9] Batch [500]	Speed: 14494.66 samples/sec	accuracy=1.000000
INFO:root:Epoch[9] Train-accuracy=1.000000
INFO:root:Epoch[9] Time cost=4.267
INFO:root:Epoch[9] Validation-accuracy=0.994000
```

# Some Problems
1. The performance of my Inception is not so good as CNN with inception layer. 
2. There is a warning when I add inception layer
	> node 'inception', graph 'cnn_inception' size too small for label
   
   How can I fix it?