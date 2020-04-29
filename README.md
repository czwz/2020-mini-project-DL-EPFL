# mini-project-2020-DL-EPFL
- This repo contains files (shown below) for the project of the course EPFL EE-559 Deep learning.

```diff
-b master: Project 3 Mini deep-learning framework
-b futong: Project 2 Classication, weight sharing, auxiliary losses
```
## -Project 3: Mini deep-learning framework

It is the python code to implement the modules such as 
```diff
linear (fully connected layer), 
sigmoid (Tanh), 
optimizer_sgd (stochastic gradient descent),
criterion_mse (mean square error),
```
and to combine above several modules in a basic sequential structure.

More detailed description about codes are as comment in corresponding files.

- Content
```diff
-test.py: main executable
-mynn_module.py: module library to be imported
```

- Usage:
```diff
$ python test.py
```

- Expected result may be like:
```diff
epoch:  0, total loss:2.464685, number of train error:176, number of test error:106
epoch: 10, total loss:1.791855, number of train error: 29, number of test error: 39
epoch: 20, total loss:1.677674, number of train error: 22, number of test error: 36
epoch: 30, total loss:1.600664, number of train error: 22, number of test error: 29
epoch: 40, total loss:1.522310, number of train error: 17, number of test error: 27
epoch: 50, total loss:1.466746, number of train error: 15, number of test error: 26
epoch: 60, total loss:1.422140, number of train error: 14, number of test error: 26
epoch: 70, total loss:1.379358, number of train error: 12, number of test error: 30
epoch: 80, total loss:1.345212, number of train error: 13, number of test error: 32
epoch: 90, total loss:1.310656, number of train error: 11, number of test error: 32
epoch:100, total loss:1.290855, number of train error: 12, number of test error: 31
```
