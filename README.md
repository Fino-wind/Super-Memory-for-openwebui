# Super-Memory-for-openwebui
作者：南风的AI进化日记。知乎，b站，抖音同号

超级记忆助手，专为中文用户优化。

1.自动分析并且存储记忆，分为fact模式（即直接提取需要记忆的事实写入记忆）和summary模式（以ai自己的视角总结记忆写入）。

2.为每一条新存入的记忆，都打上一个精确到分钟的时间戳，让ai知道每段记忆记住的时间。

3.定期后台自动生成记忆摘要，它在后台像一个 **“深思熟虑的图书管理员”** ，按照您设定的时间（比如每2小时），不打扰您正常聊天，主动巡视整个记忆库。它会把所有关于同一主题的零散记忆（比如十几条关于咖啡的记录）找出来，用LLM提炼成一条高质量的精华摘要，然后把所有旧的、零散的记录全部清理掉。
比如第一天写入记忆：用户喜欢喝拿铁咖啡，第二天写入记忆：用户喜欢在8点喝咖啡。后台会自动总结成一条精华摘要：用户喜欢在8点喝咖啡，最喜欢的咖啡是拿铁

4.拥有查重功能，不会重复记忆已有的事实。 


5. 智能事实整合自动更新记忆， **工作方式**：当您说出与过去记忆矛盾的新话时，它会立刻被激活，像一个 **“反应迅速的法官”** ，当场断案，删掉旧的错误记忆，存入新的正确记忆。
* **它的价值**：保证了记忆库的**实时准确性**。

6.全面的性能数据：完整展示每一次对话的首字时间显示、总耗时、AI的生成速度 (TPS) 和输出的Token数

7.记忆的添加，更新状态也会在面板数据那里显示
<img width="383" alt="{4A894AEB-6AEC-445A-850F-87A6650BFA3B}" src="https://github.com/user-attachments/assets/eea4b152-fa35-4273-bd6d-fbf7a3fa9071" />

使用方法：
1.openwebui社区链接导入：https://openwebui.com/f/linyou/super_memory
2.或者您可以点击代码库里面的function，复制，导入到openwebui的函数


