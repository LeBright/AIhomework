# Homework:Use DNN to Classify Pictures on CIFAR10
My homework is using DNN to classify pictures on CIFAR10. I turned two hyper-parameters and analyzed their effects on my model's performance. 
## The structure of my DNN
My DNN is based on VGG-16 in Pytorch.Its training set and test set is CIFAR10. It has 13 convolutional Layers. Instead of using 3 
full connection layers in VGG-16, I only use 1 full connection layer. 
The following table shows the details of the CNN.
Layer Name|Parameter|Output
---|:--:|---:
Dataset|-|32×32
Convolution|64,(3,3)|64×32×32
Convolution|64,(3,3)|64×32×32
Maxpooling|(2,2)|64×16×16
Convolution|128,(3,3)|128×16×16
Convolution|128,(3,3)|128×16×16
Maxpooling|(2,2)|128×8×8
Convolution|256,(3,3)|256×8×8
Convolution|256,(3,3)|256×8×8
Convolution|256,(3,3)|256×8×8
Maxpooling|(2,2)|256×4×4
Convolution|512,(3,3)|512×4×4
Convolution|512,(3,3)|512×4×4
Convolution|512,(3,3)|512×4×4
Maxpooling|(2,2)|512×2×2
Convolution|512,(3,3)|512×2×2
Convolution|512,(3,3)|512×2×2
Convolution|512,(3,3)|512×2×2
Maxpooling|(2,2)|512×1×1
Avgpooling|(2,2)|512×1×1
Full Connection|-|10
## Turing Hyper-parameter and Analyzing the Effect
I decided to turn the learning rate and the epoch of my DNN. \
Firstly, I set the epoch to 20 and turned the learning rate to 0.1, 0.01, 0.001, 0.0001 and 0.00001 to 
find the best learning rate. The following table shows the results.
Learning Rate|Accuracy(test)
---|:--:
0.1|53.75%
0.01|86.35%
0.001|88.26%
0.0001|87.68%
0.00001|76.78%

From the table, I found that the accuracy increased first and then decreased as the learning rate decreased. I thought that that was because I set the IR too large at 
the beginning and set the IR too small in the end. The large IR made the model unable to converge and hover around the optimal value. The small IR made the model 
converge so slowly that the 20 epochs was obviously not enough. Just as I learned that the IR should be set neither too large nor too small, I set the IR to 0.001 
which led to the highest accuracy.\
Secondly, I set the learning rate to 0.001 and turned the epochs to 20, 25, 30, 35, 40 to find the best epoch. The following table shows the results.
Epoch|Accuracy(test)
---|:--:
20|88.26%
25|90.95%
30|91.35%
35|91.51%
40|91.25%

From the table, I found that the accuracy increased as the epoch increased. However, the increasing of the accuracy was not obviously when the epoch were enough. 
That was because the model had converged. No matter how large the epoch I set, the accuracy of the model would not increase significantly. Therefore set the epoch too
large would be a waste of time.
## Conclusion
From this experience, I learned that we should set the IR and epoch neither too large nor too small. 
Finally, I decided to set IR to 0.001 and epoch to 35 and the final accuracy on CIFAR10 test dataset was 91.51%. 