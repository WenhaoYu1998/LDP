## DRL Navigation

### 1. 准备
安装ubuntu20.04

[安装 ROS](https://git.ustc.edu.cn/drl_navigation/startup/-/blob/master/%E7%B3%BB%E7%BB%9F%E7%8E%AF%E5%A2%83%E9%83%A8%E7%BD%B2/ubuntu20.04startup.md)，然后编译：
```
sudo apt install ros-noetic-driver-base ros-noetic-joy ros-noetic-navigation libedit-dev
sudo apt install python-scipy python-joblib python-sklearn python-pandas
```

## 相关论文
1. [Sensors](https://www.mdpi.com/1424-8220/20/17/4836), [[PDF]](https://cgdsss.gitee.io/cgd/pdf/sensors-20-04836.pdf)
```
@Article{chen2020distributed,
title = {Distributed Non-Communicating Multi-Robot Collision Avoidance via Map-Based Deep Reinforcement Learning},
author = {Chen, Guangda and Yao, Shunyi and Ma, Jun and Pan, Lifan and Chen, Yu'an and Xu, Pei and Ji, Jianmin and Chen, Xiaoping},
journal = {Sensors},
volume = {20},
number = {17},
pages = {4836},
year = {2020},
publisher = {Multidisciplinary Digital Publishing Institute},
doi = {10.3390/s20174836},
url = {https://www.mdpi.com/1424-8220/20/17/4836}
}
```
2. [ICNSC 2020](http://www.icnsc2020.org/), [[PDF]](https://cgdsss.gitee.io/cgd/pdf/ICNSC_2020_paper_11.pdf), [[Best Student Paper Award]](https://cgdsss.gitee.io/cgd/pdf/ICNSC2020Award.pdf)
```
@InProceedings{chen2020robot,
title = {Robot Navigation with Map-Based Deep Reinforcement Learning},
author = {Chen, Guangda and Pan, Lifan and Chen, Yu'an and Xu, Pei and Wang, Zhiqiang and Wu, Peichen and Ji, Jianmin and Chen, Xiaoping},
year = {2020},
pages = {1-6},
address = {Nanjing, China},
doi = {10.1109/ICNSC48988.2020.9238090},
organization = {IEEE}
}
```