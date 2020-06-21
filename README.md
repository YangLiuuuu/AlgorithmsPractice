
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

> 给你n根火柴棍，你可以拼出多少个形如“A+B=C”的等式？等式中的A、B、C是用火柴棍拼出的整数（若该数非零，则最高位不能是0）。用火柴棍拼数字0-9的拼法如图所示：
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
---
---
[验证回文串](https://leetcode-cn.com/problems/valid-palindrome/)

> 给定一个字符串，验证它是否是回文串，只考虑字母和数字字符，可以忽略字母的大小写。
> 
> 说明：本题中，我们将空字符串定义为有效的回文串。

示例：
```
输入: "A man, a plan, a canal: Panama"
输出: true

输入: "race a car"
输出: false
```
代码(python3)
```python
class Solution:
    def isPalindrome(self, s: str) -> bool:
        s=s.lower()
        i,j=0,len(s)-1
        while i<j:
            while i<j and not (s[i]>='a' and s[i]<='z') and not(s[i]>='0' and s[i]<='9'):
                i+=1
            while i<j and not (s[j]>='a' and s[j]<='z') and not(s[j]>='0' and s[j]<='9'):
                j-=1
            if i>=j:
                break
            if s[i]!=s[j]:
                return False
            i+=1
            j-=1
        return True
```

863 [二叉树中所有距离为 K 的结点](https://leetcode-cn.com/problems/all-nodes-distance-k-in-binary-tree/)

> 给定一个二叉树（具有根结点 root）， 一个目标结点 target ，和一个整数值 K 。
返回到目标结点 target 距离为 K 的所有结点的值的列表。 答案可以以任何顺序返回。

示例:
```
输入：root = [3,5,1,6,2,0,8,null,null,7,4], target = 5, K = 2
输出：[7,4,1]
解释：
所求结点为与目标结点（值为 5）距离为 2 的结点，
值分别为 7，4，以及 1
```
![](https://s3-lc-upload.s3.amazonaws.com/uploads/2018/06/28/sketch0.png)

- 给定的树是非空的。
- 树上的每个结点都具有唯一的值 0 <= node.val <= 500 。
- 目标结点 target 是树上的结点。
- 0 <= K <= 1000.

代码(c++)
```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    map<int,TreeNode*>parents;
    void findTarget(TreeNode *root,TreeNode *parent,TreeNode *target){
        if(root==NULL){
            return;
        }
        parents[root->val] = parent;
        if(root==target)return;
        findTarget(root->left,root,target);
        findTarget(root->right,root,target);
    }
    vector<int> distanceK(TreeNode* root, TreeNode* target, int K) {
	   /**
        *把树转换为图，然后进行广度优先遍历
        *由于每个结点的子节点可以找到，转换过程中我们只需记录每个结点的父节点即可，
		*可以用map记录，以结点值为键，父结点指针为值记录父子关系。
        *BFS从target结点开始，进行BFS时先找到当前结点的左右子节点，再到map中找到它的父亲，如果父亲还未遍历，则入队
        */
        set<int>book;//记录结点是否已经遍历过
        findTarget(root,NULL,target);
        vector<int>res;
        queue<TreeNode*>q;
        q.push(target);
        book.insert(target->val);
        int dis = 0;//记录距离
        while(!q.empty()){
            int size = q.size();
            if(dis==K)break;
            for(int i=0;i<size;i++){
                TreeNode *t  = q.front();
                q.pop();
                if(!t)continue;
                if(t->left && book.count(t->left->val)==0){//找左孩子
                    book.insert(t->left->val);
                    q.push(t->left);
                }
                if(t->right && book.count(t->right->val)==0){//找右孩子
                    book.insert(t->right->val);
                    q.push(t->right);
                }
                if(parents.count(t->val)){//找父亲
                    TreeNode *p = parents[t->val];
                    if(p&&book.count(p->val)==0){
                        book.insert(p->val);
                        q.push(p);
                    }
                }
            }
            dis++;
        }
        while(!q.empty()){
            res.push_back(q.front()->val);
            q.pop();
        }
        return res;
    }
};
```
---
---
[三数之和的多种可能](https://leetcode-cn.com/problems/3sum-with-multiplicity/)

> 给定一个整数数组 A，以及一个整数 target 作为目标值，返回满足 i < j < k 且 A[i] + A[j] + A[k] == target 的元组 i, j, k 的数量。
由于结果会非常大，请返回 结果除以 10^9 + 7 的余数。

示例：
```
输入：A = [1,1,2,2,3,3,4,4,5,5], target = 8
输出：20
解释：
按值枚举（A[i]，A[j]，A[k]）：
(1, 2, 5) 出现 8 次；
(1, 3, 4) 出现 8 次；
(2, 2, 4) 出现 2 次；
(2, 3, 3) 出现 2 次。

输入：A = [1,1,2,2,2,2], target = 5
输出：12
解释：
A[i] = 1，A[j] = A[k] = 2 出现 12 次：
我们从 [1,1] 中选择一个 1，有 2 种情况，
从 [2,2,2,2] 中选出两个 2，有 6 种情况。
```
- 3 <= A.length <= 3000
- 0 <= A[i] <= 100
- 0 <= target <= 300

代码(c++)
```c++
class Solution {
public:
    int threeSumMulti(vector<int>& A, int target) {
        /*
        由于A[i]在0-100之间，可以用数组记录0-100之间的数字出现次数，然后选出数字之和为target数字进行组合
        每次确定一个数字i,再寻找另外两个数字，这两个数字可以用双指针寻找，复杂度压缩为o(n)。
        外部复杂度为o(n),总复杂度为O(n^2)
        */
        int mod = 1e9+7,midx = 0,t;
        long res=0;
        int a[101] = {0};
        for(int i=0;i<A.size();i++){
            a[A[i]]+=1;
            midx = max(midx,A[i]);
        }
        for(int i=0;i<=target/2+1&&i<=100;i++){
            if(a[i]<=0) continue;
            if(i*3==target){
                if(a[i]>=3){
                    res=(res+(long)a[i]*(a[i]-1)*(a[i]-2)/6)%mod;
                }
            }else if(a[i]>=2 && target-i*2>=0 && target-i*2<=100 && a[target-i*2]>0){
                res = (res+a[i]*(a[i]-1)/2*a[target-i*2])%mod;
            }
            int left = i+1,right = midx;
            while(left<right){
                t = i+left+right;
                if(t>target){
                    right--;
                }else if(t<target){
                    left++;
                }else{
                    res = (res+a[i]*a[left]*a[right])%mod;
                    left++;
                }
            }
        }
        return res;
    }
};
```
---
---
[爱生气的书店老板](https://leetcode-cn.com/problems/grumpy-bookstore-owner/)

> 今天，书店老板有一家店打算试营业 customers.length 分钟。每分钟都有一些顾客（customers[i]）会进入书店，所有这些顾客都会在那一分钟结束后离开。
在某些时候，书店老板会生气。 如果书店老板在第 i 分钟生气，那么 grumpy[i] = 1，否则 grumpy[i] = 0。 当书店老板生气时，那一分钟的顾客就会不满意，不生气则他们是满意的。
书店老板知道一个秘密技巧，能抑制自己的情绪，可以让自己连续 X 分钟不生气，但却只能使用一次。
请你返回这一天营业下来，最多有多少客户能够感到满意的数量。

示例：
```
输入：customers = [1,0,1,2,1,1,7,5], grumpy = [0,1,0,1,0,1,0,1], X = 3
输出：16
解释：
书店老板在最后 3 分钟保持冷静。
感到满意的最大客户数量 = 1 + 1 + 1 + 1 + 7 + 5 = 16.
```
代码(python3)
```python
class Solution:
    '''
    先计算老板在不能控制情绪时感到满意的顾客数量，然后用滑动窗口计算控制情绪后最大能增加多少满意的顾客
    '''
    def maxSatisfied(self, customers: List[int], grumpy: List[int], X: int) -> int:
        n = len(grumpy)
        c = 0
        add = 0
        for i in range(0,n):
            c += customers[i] if grumpy[i]==0 else 0
        i,j=0,0
        while j<X:
            add += customers[j] if grumpy[j]==1 else 0
            j+=1
        t = add
        while j<n:
            t -= customers[i] if grumpy[i]==1 else 0
            i+=1
            t += customers[j] if grumpy[j]==1 else 0
            j+=1 
            add = max(add,t)
        return c+add
```
---
---
[从根到叶的二进制数之和](https://leetcode-cn.com/problems/sum-of-root-to-leaf-binary-numbers/)

> 给出一棵二叉树，其上每个结点的值都是 0 或 1 。每一条从根到叶的路径都代表一个从最高有效位开始的二进制数。例如，如果路径为 0 -> 1 -> 1 -> 0 -> 1，那么它表示二进制数 01101，也就是 13 。
对树上的每一片叶子，我们都要找出从根到该叶子的路径所表示的数字。
以 10^9 + 7 为模，返回这些数字之和。

示例:
![enter description here](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/04/05/sum-of-root-to-leaf-binary-numbers.png)
```
输入：[1,0,1,0,1,0,1]
输出：22
解释：(100) + (101) + (110) + (111) = 4 + 5 + 6 + 7 = 22
```
代码(c++)
```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    int mod = 1e9+7;
    int res=0;
    void fun(TreeNode *root,int val){
        val = val*2+root->val;
        if(root->left==NULL && root->right==NULL){
            res = (res+val)%mod;
            return;
        }
        if(root->left)fun(root->left,val);
        if(root->right)fun(root->right,val);
    }
    int sumRootToLeaf(TreeNode* root) {
        fun(root,0);
        return res;
    }
};
```
---
---
[驼峰式匹配](https://leetcode-cn.com/problems/camelcase-matching/)

> 如果我们可以将小写字母插入模式串 pattern 得到待查询项 query，那么待查询项与给定模式串匹配。（我们可以在任何位置插入每个字符，也可以插入 0 个字符。）
给定待查询列表 queries，和模式串 pattern，返回由布尔值组成的答案列表 answer。只有在待查项 queries[i] 与模式串 pattern 匹配时， answer[i] 才为 true，否则为 false。

示例:
```
输入：queries = ["FooBar","FooBarTest","FootBall","FrameBuffer","ForceFeedBack"], pattern = "FB"
输出：[true,false,true,true,false]
示例：
"FooBar" 可以这样生成："F" + "oo" + "B" + "ar"。
"FootBall" 可以这样生成："F" + "oot" + "B" + "all".
"FrameBuffer" 可以这样生成："F" + "rame" + "B" + "uffer".


输入：queries = ["FooBar","FooBarTest","FootBall","FrameBuffer","ForceFeedBack"], pattern = "FoBa"
输出：[true,false,true,false,false]
解释：
"FooBar" 可以这样生成："Fo" + "o" + "Ba" + "r".
"FootBall" 可以这样生成："Fo" + "ot" + "Ba" + "ll".
```
代码(python3)
```python
class Solution:
    def camelMatch(self, queries: List[str], pattern: str) -> List[bool]:
        n,plen = len(queries),len(pattern)
        res = [False]*n
        for i in range(n):
            j,k,f = 0,0,1
            while j<plen and k<len(queries[i]):
                while k<len(queries[i]) and queries[i][k]!=pattern[j]:
                    if pattern[j]>='a' and pattern[j]<='z':
                        if queries[i][k]>='A' and queries[i][k]<='Z':
                            f=0
                            break
                    elif pattern[j]>='A' and pattern[j]<='Z':
                        if queries[i][k]>='A' and queries[i][k]<='Z' and queries[i][k]!=pattern[j]:
                            f=0
                            break
                    k+=1
                if k>=len(queries[i]) or queries[i][k]!=pattern[j]:
                    f = 0
                    break
                j+=1
                k+=1
            while k<len(queries[i]):
                if queries[i][k]>='A' and queries[i][k]<='Z':
                    f=0
                    break
                k+=1
            if f==1:
                res[i] = True
        return res    
```





