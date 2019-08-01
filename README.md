# HumanPose
人体骨骼14点调参心得

原始工程来源于https://github.com/edvardHua/PoseEstimationForMobile

但是其中有些BUG，我略作修改，如果你没法跑起来原工程，可以把我的training文件夹弄过去替换掉原工程的。

# 150M计算量调参心得（双核A53约50ms）

**（1）stage_num=3，分辨率192x192调参阶段**

stage_num=3时，网络设计捉襟见肘，channel和kernel都不大。

这一阶段包括zq1_cpm, zq7_cpm, zq8_cpm, zq9_cpm, zq10_cpm。

PCKh0.2从62.4提高到64.81, 模型结构请查阅代码, 精度请查阅xls。

**（2）stage_num=2，分辨率192x192调参阶段**

stage_num=2时，网络设计可以活动的地方较大，随便设计一个里面超过先前stage_num=3

这一阶段baseline是zq11_cpm， PCKh0.2达到65.98， 越改越差的包括zq12_cpm, zq13_cpm, zq14_cpm, zq15_cpm。

调参过程包括zq22_cpm，（未完待续）

模型结构请查阅代码，精度请查阅xls

**（3）stage_num=2,分辨率224x224调参阶段**

考虑到heatmap分辨率太小会导致无论无何都不可能太准，尝试加大分辨率。但不得不缩减channel和kernel。

这一阶段baseline是zq16_cpm （模型设计思路来源于zq11_cpm）， PCKh0.2达到65.64， 越改越差的包括zq17_cpm, zq18_cpm, zq19_cpm, zq20_cpm

第一次改好的结构是zq21_cpm, PCKh0.2达到66.73

