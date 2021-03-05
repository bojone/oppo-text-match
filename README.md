# oppo-text-match
小布助手对话短文本语义匹配的一个baseline

## 模型

参考：https://kexue.fm/archives/8213

base版本线下大概0.952，线上0.866（单模型，没做K-flod融合）。

## 训练

测试环境：tensorflow 1.15 + keras 2.3.1 + bert4keras 0.10.0

跑完100epoch可能6小时左右（3090，建议跑完）

## 预测

```python
from baseline import *
predict_to_file('result.csv')
```
然后`zip result.zip result.csv`，最后把result.zip提交即可。

## 感谢

感谢主办方对本baseline的肯定～

## 交流

- 比赛交流群：QQ群753413531
- 科学空间交流：QQ群808623966，微信群请加机器人微信号spaces_ac_cn
