
816 [模糊坐标](https://leetcode-cn.com/problems/ambiguous-coordinates/)

> 我们有一些二维坐标，如 "(1, 3)" 或 "(2, 0.5)"，然后我们移除所有逗号，小数点和空格，得到一个字符串S。返回所有可能的原始字符串到一个列表中。
原始的坐标表示法不会存在多余的零，所以不会出现类似于"00", "0.0", "0.00", "1.0", "001", "00.01"或一些其他更小的数来表示坐标。此外，一个小数点前至少存在一个数，所以也不会出现“.1”形式的数字。
最后返回的列表可以是任意顺序的。而且注意返回的两个数字中间（逗号之后）都有一个空格。

示例:
```
输入: "(123)"
输出: ["(1, 23)", "(12, 3)", "(1.2, 3)", "(1, 2.3)"]


输入: "(00011)"
输出:  ["(0.001, 1)", "(0, 0.011)"]
解释: 
0.0, 00, 0001 或 00.01 是不被允许的。

输入: "(0123)"
输出: ["(0, 123)", "(0, 12.3)", "(0, 1.23)", "(0.1, 23)", "(0.1, 2.3)", "(0.12, 3)"]

输入: "(100)"
输出: [(10, 0)]
解释: 
1.0 是不被允许的。
```

代码(python3)
```python
class Solution:
    def fun(self,s):
        n = len(s)
        res = []
        if n==1:
            res.append(s)
            return res
        if s[0]=='0':
            if s[n-1]=='0':
                return []
            else:
                res.append(s[0]+'.'+s[1:])
        else:
            res.append(s)
            if s[n-1]=='0':
                return res
            else:
                for i in range(len(s)-1):
                    res.append(s[:i+1]+'.'+s[i+1:])
        return res
    def ambiguousCoordinates(self, S: str) -> List[str]:
        '''
        先用逗号将数字分为两部分，再分别判断两部分可以拆成哪些数字
        如果数字开头是0，而且结尾也是0，那么这个数字无效
        如果数字开头是0，结尾不是0，它只有一种拆解方法，即0.xxx
        如果开头数字不是0，它本身就是一种拆解方法，然后看末尾是不是0，如果末尾是0，那么它不能再拆解，否则，
        每个数字之间都可以添加小数点
        '''
        res = []
        for i in range(2,len(S)-1):
            r1 = self.fun(S[1:i])
            r2 = self.fun(S[i:len(S)-1])
            if not r1 or not r2:
                continue
            for s1 in r1:
                for s2 in r2:
                    res.append('('+s1+', '+s2+')')
        return res
```
---
---
[火柴棒等式](https://www.luogu.com.cn/problem/P1149)

> 给你n根火柴棍，你可以拼出多少个形如“A+B=CA+B=C”的等式？等式中的AA、BB、CC是用火柴棍拼出的整数（若该数非零，则最高位不能是00）。用火柴棍拼数字0-90−9的拼法如图所示：
![enter description here](https://cdn.luogu.com.cn/upload/pic/49.png)
注意：
加号与等号各自需要两根火柴棍
如果A≠B,则A+B=C与B+A=C视为不同的等式(A,B,C>=0)
n根火柴棍必须全部用上

输入一个整数n(n<=24)
输出一个数表示能拼成的不同等式的数目。

如输入18，输出9

9个等式为：
```
0+4=4
0+11=11
1+10=11
2+2=4
2+7=9
4+0=4
7+2=9
10+1=11
11+0=11
```

代码(python3)
```python
'''
因为输入的n小于24，数据量不是很大，可以摆出的最大数字不超过1000.穷举出0-2000的表示需要的火柴棒数，然后查找所有满足条件等式的种数
'''
n = int(input())
n-=4
book = [6,2,5,5,4,5,6,3,7,6]
a = [0]*2001
a[0]=6
for i in range(1,2001):
    j = i
    while j>0:
        a[i]+=book[j%10]
        j//=10
    #print(a[i],end=' ')
res=0
for i in range(1001):
    for j in range(1001):
        if a[i]+a[j]+a[i+j]==n:
            res+=1
print(res)

```



