# 大模型学习与 CS224N 笔记

本工作空间收录大模型基础术语笔记、斯坦福 CS224N 课程学习笔记及 NumPy 作业练习。

## 内容

- `outputs/`：可阅读的 Markdown 笔记与 Jupyter Notebook 练习。
- `work/learning-notes/`：持续收集的笔记材料。
- `outputs/练习模板/CS224N-A1-前两题-简化与原版.ipynb`：公开空白练习，每题分为自编简化用例与官方用例，保留填写区域及思考题。
- 原有个人练习文件仅在本地保留，由 `.gitignore` 排除。后续作答可放在 `outputs/个人作答/`，此目录也不上传。

## 使用练习

使用 Python 3 与 NumPy，在 Jupyter 中打开 Notebook。按初始化、函数定义、检查用例的顺序运行；函数修改后需重新执行其单元格。

```powershell
python -m pip install numpy notebook
python -m notebook
```

IMDB 实验为可选部分，另需 `datasets` 和联网下载；前两题的小语料检查可以离线进行。

## 课程来源

[Stanford CS224N Spring 2024](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/)

笔记与练习包含中文解释和自编算例，具体官方资料来源见各文件。Notebook 可能包含个人练习实现，不代表官方解答。
