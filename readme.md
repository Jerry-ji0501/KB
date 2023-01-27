# 备注

split_data -- 从脑电信号进行窗口划分
loadData -- 对模型输入数据进行加载(脑电特征、ZPI、节点嵌入)
CNN -- 对ZPI进行卷积提取特征
ZGCN -- 进行时空卷积
GRU -- 对时空卷积特征进行序列操作
train -- 对整个模型进行串联
zigzagtools -- ZPI计算工具
BrainGraphZPI -- 计算绘制ZPI图
Add_Windows -- 加窗