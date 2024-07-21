# file structure
```
|-log/ # 存放日志文件
    |-
|-model/ # 存放模型文件
    |-
data.py # dataloader
train.py # 训练文件
utils.py
Transformer_EncDec.py # Encoder Layer
SelfAttention_Family.py # Self-Attention
iTransformer_temp.py # temp模型
iTransformer_wind.py # wind模型
Dec.py # decoder 文件
Embed.py # 编码文件
config.json # 配置文件
index.py # submit predict file
```

# file detail
 + data.py
    + root_path: 数据集存放位置
    + mode: train or test or predict
    + data_split: None
    + global_wind: 矢量风速计算成标量风速
    + wind_direction: 计算风向
    + MLM: 预训练token
    + MLM_mask: 预训练token是否需要MLM_mask
    + task: temp or wind
    + temp_add: global数据里面的temp对齐到temp

 + config.json
    + seed: 随机种子
    + batch_size: 批处理大小
    + num_workers: dataloader num workers
    + lr: learning rate
    + weight_decay: None
    + gpu: 0 or 1 or 2 if multi gpus to use single gpu
    + epochs: train epochs
    + early_stop: early stop
    + Transformer
        + None
    + wind / temp
        + global_wind: 是否使用矢量风速(1 代表使用，0代表不使用)
        + wind_direction: 风向
        + MLM: 预训练token
        + MLM_mask: 预训练token的mask
        + KL_Loss: 训练时是否使用KL散度（0代表不使用，非0代表使用）
        + temp_add: global里面的temp是否对齐到temp
    + temp_MLM: 预训练参数
        + None

# The best score: 0.991
wind和temp分开训练，wind模型采用31，temp模型采用111。wind首先采用三个空间注意力机制，然后采用一个全注意力。temp采用一个全注意力提取信息然后采用一个空间注意力，在采用一个全注意力。wind采用线性解码器，temp采用CNN多周期性解码器。

wind和temp收敛速度不一致，wind大约3轮，temp大约8轮。

其中，temp上使用了temp_add特征，提升幅度小于0.001，不排除误差影响。可以考虑去掉该特征。