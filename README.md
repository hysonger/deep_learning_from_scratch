Deep Learning From Scratch (Self-written Code Version)
-------------
《深度学习入门：基于Python的理论与实现》自写版本代码

其实也不算完全自写……没有前三章的练习，主要是4-8章的四个模型（也没有数值微分部分，谁闲得没事写这种老掉牙玩意啊，我才不会说是因为我没做出来）。

修改主要有这么几点：

1. 原书的代码风格有些前后不一致，我把风格基本上归一化了，比如ReLU和Dropout层
2. 把一些关键逻辑在理解的基础上改了改
3. 代码和类的框架与附赠代码有区别。所有的网络类都继承自同一个Net，方法实现写在基类里，主要区别只在self.layers的内容不同；没有Trainer类，每个Layer类自己内部保管权重(param)和梯度(grad)，构造时指定一个Optimizer类及其参数，更新权重时对外暴露一个update_params()方法供Net调用；
4. 把numpy替换成了cupy以加快运算。如没有N卡或不想用cupy者，请全文搜索替换import cupy => import numpy并删除dataset/mnist.pkl重新生成数据集
5. 更详细的注释！