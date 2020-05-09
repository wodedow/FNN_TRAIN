# FNN_TRAIN
前馈神经网络：训练模块

`Struct.h` 是结构体定义和初始化的头文件

主体函数 `main` 定义了神经网络的结构
```c/c++
  int L = 3;  //神经网络的层数
  int m[] = { img_size, 300,50, classes };  //每一层的结点数：m[1:]
  char optimizer[] = "Adam";
  bool first = true;  //是否初次运行网络

  FNN Net;
  Init_Network_FNN(Net, L, m, batch_sizes);
```
主体函数 `main` 调用 `FNNTrain` 函数运行训练函数
1. `createDatas` : 初始化权重矩阵与偏置（参数）
2. `readDataFromMnist_train` : 从数据集中读取数据，并返回文件指针
3. `forward` : 前向传播更新各层的输入输出
4. `backward` : 反向传播得到各参数的偏导
5. `parameters_update_Adam` : 选择优化方法 `Adam` 更新参数

`readDataFromMnist_train_con` : 从数据集中继续读取数据，并重复 `3-5`，直到满足停止条件。

数据的存储
* `save_weight_arrays` : 存储权重矩阵 `w` 到TXT文件
* `save_bias` : 存储偏置 `b` 
* `save_offset` : 存储图像与标签文件的文件指针，方便下次使用

***
程序主体上是采用C编写，实际上还是使用了C++的一些性质，如 `new` 函数。

未使用面向对象编程
