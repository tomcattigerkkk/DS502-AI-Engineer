# MXNET-Week1 part2 

Homework for Week1 part2. 

The structures of my network are illustrated in PDFs, **MLP.pdf**, **CNN.pdf**,   
 **CNN_inception.pdf**.

# Some comparisons 

## Short Result
 	|Comparison              | MLP | CNN | CNN with Inception |
	
	|Acc(5 epoch)            | - | - | .99386 |

	|CPU Epoch time          | - | - | - |
	
	|GPU Epoch Time(optional)| - | - | 14494.66 samples/sec |

## Details of test results 

### MLP (CPU)
```text
```

### MLP (GPU)
```text
```

### CNN (CPU)
```text

```

### CNN (GPU)
```text

```

### CNN with Inception Layer (CPU)
```text

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