
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
---
---
[视频拼接](https://leetcode-cn.com/problems/video-stitching/)

>你将会获得一系列视频片段，这些片段来自于一项持续时长为 T 秒的体育赛事。这些片段可能有所重叠，也可能长度不一。
视频片段 clips[i] 都用区间进行表示：开始于 clips[i][0] 并于 clips[i][1] 结束。我们甚至可以对这些片段自由地再剪辑，例如片段 [0, 7] 可以剪切成 [0, 1] + [1, 3] + [3, 7] 三部分。
我们需要将这些片段进行再剪辑，并将剪辑后的内容拼接成覆盖整个运动过程的片段（[0, T]）。返回所需片段的最小数目，如果无法完成该任务，则返回 -1 。

示例：
```
输入：clips = [[0,2],[4,6],[8,10],[1,9],[1,5],[5,9]], T = 10
输出：3
解释：
我们选中 [0,2], [8,10], [1,9] 这三个片段。
然后，按下面的方案重制比赛片段：
将 [1,9] 再剪辑为 [1,2] + [2,8] + [8,9] 。
现在我们手上有 [0,2] + [2,8] + [8,10]，而这些涵盖了整场比赛 [0, 10]。


输入：clips = [[0,1],[1,2]], T = 5
输出：-1
解释：
我们无法只用 [0,1] 和 [0,2] 覆盖 [0,5] 的整个过程。


输入：clips = [[0,4],[2,8]], T = 5
输出：2
解释：
注意，你可能录制超过比赛结束时间的视频。
```
代码(python3)
```python
class Solution:
    def cmp(self, lst1, lst2):
        if lst1[0]<lst2[0]:
            return -1
        elif lst1[0]>lst2[0]:
            return 1
        elif lst1[1]<lst2[1]:
            return 1
        else:
            return -1
    def videoStitching(self, clips: List[List[int]], T: int) -> int:
        '''
        将区间按照左端值排序，然后贪心选择使得区间连续而且能够达到的最右区间值
        排序O(n log n),选择O(n^2),总复杂度O(n^2)
        '''
        clips = sorted(clips,key=functools.cmp_to_key(self.cmp))
        if clips[0][0]!=0:
            return -1
        left,right = clips[0][0],clips[0][1]
        c,i,le = 1,1,len(clips)
        while right<T:
            tright = right
            idx = -1
            for j in range(i,le):
                if clips[j][0]>right:
                    break
                if clips[j][0]<=right and clips[j][1]>=tright:
                    idx = j
                    tright = clips[j][1]
            if idx==-1:
                return -1
            c+=1
            left = clips[idx][0]
            right = clips[idx][1]
            i = idx+1
        if right>=T:
            return c
        return -1
```
---
---
[节点与其祖先之间的最大差值](https://leetcode-cn.com/problems/maximum-difference-between-node-and-ancestor/)

> 给定二叉树的根节点 root，找出存在于不同节点 A 和 B 之间的最大值 V，其中 V = |A.val - B.val|，且 A 是 B 的祖先。
（如果 A 的任何子节点之一为 B，或者 A 的任何子节点是 B 的祖先，那么我们认为 A 是 B 的祖先）

示例:
![enter description here](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/04/12/2whqcep.jpg)
```
输入：[8,3,10,1,6,null,14,null,null,4,7,13]
输出：7
解释： 
我们有大量的节点与其祖先的差值，其中一些如下：
|8 - 3| = 5
|3 - 7| = 4
|8 - 1| = 7
|10 - 13| = 3
在所有可能的差值中，最大值 7 由 |8 - 1| = 7 得出。
```
代码(python3)
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def dfs(self,root,maxv,minv):
        if not root:
            return 0
        maxv = max(maxv,root.val)
        minv = min(minv,root.val)
        if not root.left and not root.right:
            return maxv-minv
        return max(self.dfs(root.left,maxv,minv),self.dfs(root.right,maxv,minv))
        
    def maxAncestorDiff(self, root: TreeNode) -> int:
        return max(self.dfs(root.left,root.val,root.val),self.dfs(root.right,root.val,root.val))        

```
---
---
[二进制求和](https://leetcode-cn.com/problems/add-binary/)

>给你两个二进制字符串，返回它们的和（用二进制表示）。
输入为 非空 字符串且只包含数字 1 和 0。

示例 1:
```
输入: a = "11", b = "1"
输出: "100"
```
代码(java)
```java
class Solution {
    public String addBinary(String a, String b) {
        StringBuilder sb = new StringBuilder();
        int len1 = a.length(),len2 = b.length();
        int i = len1-1,j = len2-1,n1,n2,carry=0;
        while(i>=0||j>=0||carry!=0){
            n1 = i>=0 ? a.charAt(i)-'0':0;
            n2 = j>=0 ? b.charAt(j)-'0':0;
            sb.append((n1 + n2 + carry) % 2);
            carry = (n1+n2+carry)/2;
            i--;
            j--;
        }
        return sb.reverse().toString();
    }
}
```
[距离顺序排列矩阵单元格](https://leetcode-cn.com/problems/matrix-cells-in-distance-order/)

>给出 R 行 C 列的矩阵，其中的单元格的整数坐标为 (r, c)，满足 0 <= r < R 且 0 <= c < C。
另外，我们在该矩阵中给出了一个坐标为 (r0, c0) 的单元格。
返回矩阵中的所有单元格的坐标，并按到 (r0, c0) 的距离从最小到最大的顺序排，其中，两单元格(r1, c1) 和 (r2, c2) 之间的距离是曼哈顿距离，|r1 - r2| + |c1 - c2|。（你可以按任何满足此条件的顺序返回答案。）

示例 :
```
输入：R = 1, C = 2, r0 = 0, c0 = 0
输出：[[0,0],[0,1]]
解释：从 (r0, c0) 到其他单元格的距离为：[0,1]


输入：R = 2, C = 2, r0 = 0, c0 = 1
输出：[[0,1],[0,0],[1,1],[1,0]]
解释：从 (r0, c0) 到其他单元格的距离为：[0,1,1,2]
[[0,1],[1,1],[0,0],[1,0]] 也会被视作正确答案。
```

代码(java)
```java
class Solution {
    class Point{
        int x,y;
        public Point(int x,int y){
            this.x = x;
            this.y = y;
        }
    }
    public int[][] allCellsDistOrder(int R, int C, int r0, int c0) {
		//简单bfs
        int[][] res = new int[R*C][2];
        int[][] book = new  int[R][C];
        int[][] dir = {{-1,0},{1,0},{0,1},{0,-1}};
        int k=0;
        Queue<Point> queue = new LinkedList<>();
        queue.offer(new Point(r0,c0));
        book[r0][c0]=1;
        while (!queue.isEmpty()){
            int size = queue.size();
            for (int i=0;i<size;i++){
                Point p = queue.poll();
                res[k][0] = p.x;
                res[k++][1] = p.y;
                for (int j=0;j<4;j++){
                    int nrow = p.x+dir[j][0],ncol=p.y+dir[j][1];
                    if (nrow>=0 && nrow<R && ncol>=0 && ncol<C){
                        if (book[nrow][ncol]==0){
                            book[nrow][ncol] = 1;
                            queue.offer(new Point(nrow,ncol));
                        }
                    }
                }
            }
        }
        return res;
    }
}
```
---
---
[最接近的三数之和](https://leetcode-cn.com/problems/3sum-closest/)

> 给定一个包括 n 个整数的数组 nums 和 一个目标值 target。找出 nums 中的三个整数，使得它们的和与 target 最接近。返回这三个数的和。假定每组输入只存在唯一答案。

示例：
```
输入：nums = [-1,2,1,-4], target = 1
输出：2
解释：与 target 最接近的和是 2 (-1 + 2 + 1 = 2) 。
```

- 3 <= nums.length <= 10^3
- -10^3 <= nums[i] <= 10^3
-  -10^4 <= target <= 10^4

代码(java)
```java
class Solution {
    public int threeSumClosest(int[] nums, int target) {
        Arrays.sort(nums);
        int res = nums[0]+nums[1]+nums[2];
        int left,right,sum;
        for (int i=0;i<nums.length;i++){
            left = i+1;
            right=nums.length-1;
            while(left<right){
                sum = nums[i]+nums[left]+nums[right];
                if(Math.abs(sum-target)<Math.abs(res-target)){
                    res = sum;
                }
                if (sum>target){
                    right--;
                }else if(sum<target){
                    left++;
                }else{
                    return target;
                }
            }
        }
        return res;
    }
}
```
---
---
[两个非重叠子数组的最大和](https://leetcode-cn.com/problems/maximum-sum-of-two-non-overlapping-subarrays/)

> 给出非负整数数组 A ，返回两个非重叠（连续）子数组中元素的最大和，子数组的长度分别为 L 和 M。（这里需要澄清的是，长为 L 的子数组可以出现在长为 M 的子数组之前或之后。）
从形式上看，返回最大的 V，而 V = (A[i] + A[i+1] + ... + A[i+L-1]) + (A[j] + A[j+1] + ... + A[j+M-1]) 并满足下列条件之一：
- 0 <= i < i + L - 1 < j < j + M - 1 < A.length, 
- 0 <= j < j + M - 1 < i < i + L - 1 < A.length

示例：
```
输入：A = [0,6,5,2,2,5,1,9,4], L = 1, M = 2
输出：20
解释：子数组的一种选择中，[9] 长度为 1，[6,5] 长度为 2。


输入：A = [3,8,1,3,2,1,8,9,0], L = 3, M = 2
输出：29
解释：子数组的一种选择中，[3,8,1] 长度为 3，[8,9] 长度为 2。


输入：A = [2,1,5,6,0,9,5,0,3,8], L = 4, M = 3
输出：31
解释：子数组的一种选择中，[5,6,0,9] 长度为 4，[0,3,8] 长度为 3。
```
- L >= 1
- M >= 1
- L + M <= A.length <= 1000
- 0 <= A[i] <= 1000

代码(java)
```java
class Solution {
    public int maxSumTwoNoOverlap(int[] A, int L, int M) {
        /**
         * 先求得数组的前缀和，然后暴力穷举所有结果
         */
        int n = A.length;
        int[] sum = new int[n+1];
        for (int i=1;i<=n;i++){
            sum[i]=sum[i-1]+A[i-1];
        }
        int res = Integer.MIN_VALUE;
        for (int i=1;i<=n-1;i++){
            for (int j=i+1;j<=n;j++){
                if (i>=L && j-i>=M){
                    res = Math.max(sum[i]-sum[i-L]+sum[j]-sum[j-M],res);
                }
                if (i>=M && j-i>=L){
                    res = Math.max(sum[i]-sum[i-M]+sum[j]-sum[j-L],res);
                }
            }
        }
        return res;
    }
}
```
---
---
139 [单词拆分](https://leetcode-cn.com/problems/word-break/)

> 给定一个非空字符串 s 和一个包含非空单词列表的字典 wordDict，判定 s 是否可以被空格拆分为一个或多个在字典中出现的单词。
拆分时可以重复使用字典中的单词。
你可以假设字典中没有重复的单词。

示例 ：
```
输入: s = "leetcode", wordDict = ["leet", "code"]
输出: true
解释: 返回 true 因为 "leetcode" 可以被拆分成 "leet code"。


输入: s = "applepenapple", wordDict = ["apple", "pen"]
输出: true
解释: 返回 true 因为 "applepenapple" 可以被拆分成 "apple pen apple"。
注意你可以重复使用字典中的单词。


输入: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
输出: false
```
代码(java)
```java
class Solution {
    public boolean wordBreak(String s, List<String> wordDict) {
        //为了加快字符串的查找速度可以将字典中的字符加入Set
        //用动态规划判断字符是否可以有字典中的单词拼接而成，dp[i]=1代表前i个字符是否可以拼接
        //dp[i] = 1 if s[j:i] in wordsDict,其中0<=j<i
        Set<String> words = new HashSet<>(wordDict);
        int[] dp = new int[s.length()+1];
        dp[0] = 1;
        for (int i=1;i<=s.length();i++){
            for (int j=0;j<i;j++){
                if (dp[j]==1&&words.contains(s.substring(j,i))){
                    dp[i]=1;
                    break;
                }
            }
        }
        return dp[s.length()]==1;
    }
}
```
---
---

1035 [不相交的线](https://leetcode-cn.com/problems/uncrossed-lines/)

> 我们在两条独立的水平线上按给定的顺序写下 A 和 B 中的整数。
现在，我们可以绘制一些连接两个数字 A[i] 和 B[j] 的直线，只要 A[i] == B[j]，且我们绘制的直线不与任何其他连线（非水平线）相交。
以这种方法绘制线条，并返回我们可以绘制的最大连线数。

![enter description here](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/04/28/142.png)
示例 
```
输入：A = [1,4,2], B = [1,2,4]
输出：2
解释：
我们可以画出两条不交叉的线，如上图所示。
我们无法画出第三条不相交的直线，因为从 A[1]=4 到 B[2]=4 的直线将与从 A[2]=2 到 B[1]=2 的直线相交。


输入：A = [2,5,1,2,5], B = [10,5,2,1,5,2]
输出：3
```
代码(java)
```java
class Solution {
    public int maxUncrossedLines(int[] A, int[] B) {
        /**
         * 可以这样想，既然连完线后他们不想交，我们可以把连线的两个数字移动对齐
         * 这样其实就是求两个串的最长公共子串，动态规划即可
         */
        int[][] dp = new int[A.length+1][B.length+1];
        for (int i=1;i<=A.length;i++){
            for (int j=1;j<=B.length;j++){
                if (A[i-1]==B[j-1]){
                    dp[i][j] = dp[i-1][j-1]+1;
                }else{
                    dp[i][j] = Math.max(dp[i-1][j],dp[i][j-1]);
                }
            }
        }
        return dp[A.length][B.length];
    }
}
```
---
---
1037[有效的回旋镖](https://leetcode-cn.com/problems/valid-boomerang/)

> 回旋镖定义为一组三个点，这些点各不相同且不在一条直线上。
给出平面上三个点组成的列表，判断这些点是否可以构成回旋镖。

示例
```
输入：[[1,1],[2,3],[3,2]]
输出：true

输入：[[1,1],[2,2],[3,3]]
输出：false
```
代码(python3)
```python
class Solution:
    def isBoomerang(self, points: List[List[int]]) -> bool:
        '''
        三个点可以得到两个向量，判断向量是否平行即可
        若向量a(x1,y1),b(x2,y2)平行，有a=kb,则x1*y2=y1*x2
        '''
        a1 = [points[1][0]-points[0][0],points[1][1]-points[0][1]]
        a2 = [points[2][0]-points[1][0],points[2][1]-points[1][1]]
        return a1[0]*a2[1]!=a1[1]*a2[0]

```








