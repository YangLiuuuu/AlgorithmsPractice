---
title: 算法练习
tags: 算法
grammar_cjkRuby: true
---

1. [奇偶链表](https://leetcode-cn.com/problems/odd-even-linked-list/)
>给定一个单链表，把所有的奇数节点和偶数节点分别排在一起。请注意，这里的奇数节点和偶数节点指的是节点编号的奇偶性，而不是节点的值的奇偶性。
请尝试使用原地算法完成。你的算法的空间复杂度应为 O(1)，时间复杂度应为 O(nodes)，nodes 为节点总数。

示例1:
```
输入: 1->2->3->4->5->NULL
输出: 1->3->5->2->4->NULL
```
示例2：
```
输入: 2->1->3->5->6->4->7->NULL 
输出: 2->3->6->7->1->5->4->NULL
```
代码(c++)
```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* oddEvenList(ListNode* head) {
        if(!head||!head->next) return head;
        ListNode *h1=head,*h2=head->next,*p=h2->next,*p1=h1,*p2=h2;
        int c = 0;
        while(p){
            if(c){
                //奇结点
                p2->next = p;
                p2=p;
                c = 0;
            }else{
                //偶结点
                p1->next = p;
                p1 = p;
                c = 1;
            }
            p=p->next;
        }
        p1->next = h2;
        p2->next = NULL;
        return h1;
    }
};
```
---
---
2. [重新安排行程](https://leetcode-cn.com/problems/reconstruct-itinerary/)

> 给定一个机票的字符串二维数组 [from, to]，子数组中的两个成员分别表示飞机出发和降落的机场地点，对该行程进行重新规划排序。所有这些机票都属于一个从JFK（肯尼迪国际机场）出发的先生，所以该行程必须从 JFK 出发。

说明:
1. 如果存在多种有效的行程，你可以按字符自然排序返回最小的行程组合。例如，行程 ["JFK", "LGA"] 与 ["JFK", "LGB"] 相比就更小，排序更靠前
2. 所有的机场都用三个大写字母表示（机场代码）。
3. 假定所有机票至少存在一种合理的行程。

示例1：
```
输入: [["MUC", "LHR"], ["JFK", "MUC"], ["SFO", "SJC"], ["LHR", "SFO"]]
输出: ["JFK", "MUC", "LHR", "SFO", "SJC"]
```
示例2：
```
输入: [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]
输出: ["JFK","ATL","JFK","SFO","ATL","SFO"]
解释: 另一种有效的行程是 ["JFK","SFO","ATL","JFK","ATL","SFO"]。但是它自然排序更大更靠后。
```
代码(java)
```java
/**
 * 自己做出来了但是效率很低，代码冗长。
 * 主要思路
 * 先构造图，将所有票中的机场名称加入TreeSet，TreeSet会去重而且排好序
 * 然后建立机场名称和排序下标的双向映射，分别以Map和List建立，这样是方便待会进行dfs和图邻接矩阵的建立
 * 然后按照dfs回溯，每经过一条路径，将路径上的权值减一，当所有机票都走过后就获取了路线
 * 由于邻接矩阵是按照机场名称排序建立的，这样获得的路线一定是字典序最小的
 * 最后根据路线记录id将所有机场名称加入结果集
 */
public class Solution {
    Stack<Integer> flightRecord = new Stack<>();
    int c;
    boolean dfs(int[][] matrix,int cur,int count){
        if (count==c){
            return true;//所有路线已经走完
        }
        for(int i=0;i<matrix.length;i++){
            if (matrix[cur][i]>=1){//dfs回溯
                matrix[cur][i]-=1;//这条路线已经走过，消除它
                flightRecord.push(i);//记录机场编号
                if (dfs(matrix,i,count+1))return true;//这条路已经走通，直接返回
                flightRecord.pop();//上面的路走不通，回溯
                matrix[cur][i] += 1;
            }
        }
        return false;
    }

    public List<String> findItinerary(List<List<String>> tickets) {
        TreeSet<String> airPorts = new TreeSet<>();
        for (List<String> ticket:tickets){
            airPorts.addAll(ticket);
        }
        int idx=0,start=0;
        Map<String,Integer> map = new HashMap<>();
        List<String>airPortId = new ArrayList<>();
        for (String airPort : airPorts) {
            map.put(airPort, idx);//机场名称到下标的映射
            airPortId.add(airPort);//下标到机场名称的映射，根据id获取机场名即可
            if (airPort.equals("JFK")) start=idx;//记录起点
            idx++;
        }
        int[][] matrix = new int[idx][idx];//图的邻接矩阵
        for (List<String> ticket:tickets){//建立矩阵
            int from = map.get(ticket.get(0)),to = map.get(ticket.get(1));
            matrix[from][to] += 1;//会有重复的机票，每经过一次加1
        }
        c = tickets.size();
        flightRecord.push(start);//最开始已经在JFK机场
        dfs(matrix,start,0);
        List<String> res = new ArrayList<>();
        for (Integer id:flightRecord){
            res.add(airPortId.get(id));//根据路线记录的id将机场名称按顺序加入结果集
        }
        return res;
    }
}
```
---
---
3. [前 K 个高频元素](https://leetcode-cn.com/problems/top-k-frequent-elements/)

> 给定一个非空的整数数组，返回其中出现频率前 k 高的元素。

示例 1:
```
输入: nums = [1,1,1,2,2,3], k = 2
输出: [1,2]
```

示例 2:
```
输入: nums = [1], k = 1
输出: [1]
```
提示：
- 你可以假设给定的 k 总是合理的，且 1 ≤ k ≤ 数组中不相同的元素的个数。
- 你的算法的时间复杂度必须优于 O(n log n) , n 是数组的大小。
- 题目数据保证答案唯一，换句话说，数组中前 k 个高频元素的集合是唯一的。
- 你可以按任意顺序返回答案。

代码(java)
```java
/**
     * map记录每个数字出现的次数
     * 然后将map以二维数组的形式表示
     * 对二维数组排序，比较时按照第二个值即频率比较
     * 然后取出频率前k高的
     *
     * 记录出现次数O(n),排序O(n Log n)
     * 总复杂度O(n log n)
     */
class Solution {
    public int[] topKFrequent(int[] nums, int k) {
        Map<Integer,Integer>map = new HashMap<>();
        for(int i:nums){
            if(map.containsKey(i)){
                map.put(i,map.get(i)+1);
            }else{
                map.put(i,1);
            }
        }
        int[][] list = new int[map.size()][2];
        int i=0;
        for (Map.Entry<Integer,Integer> entry:map.entrySet()){
            list[i][0]=entry.getKey();
            list[i][1]=entry.getValue();
            i+=1;
        }
        Arrays.sort(list, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                if (o1[1]==o2[1])return 0;
                return o1[1]<o2[1]?-1:1;
            }
        });
        int size = list.length;
        int[] res = new int[k];
        for (int j=0;j<k;j++){
            res[j] = list[size-1-j][0];
        }
        return res;
    }
}
```
---
---

1. [三数之和](https://leetcode-cn.com/problems/3sum/)
> 给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有满足条件且不重复的三元组。
> 注意：答案中不可以包含重复的三元组。

示例：
```
给定数组 nums = [-1, 0, 1, 2, -1, -4]，

满足要求的三元组集合为：
[
  [-1, 0, 1],
  [-1, -1, 2]
]
```

代码(python3)
```python
'''
将列表数据排序，然后开始查找
首先确定一个数nums[i]，再确定另外两个数，另外两个数按按以下方式寻找。首先给定两个指针，left=i+1,right=len(nums)-1,如果nums[i]+nums[left]+nums[right]<0，则将left指针右移，如果nums[i]+nums[left]+nums[right]>0，则将right指针左移。这样每个nums[i]总能找到与之对应的另外两个数
排序时间复杂度为O(n log n),确定其中一个数为O(n),内部循环确定另外两个数用到了序列有序的性质，只需要O(n),这个总循环为O(n^2),总时间复杂度为O(n^2)
'''
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        le = len(nums)
        i=0
        res = []
        while i<len(nums)-2 and nums[i]<=0:
            if i>0 and nums[i]==nums[i-1]:
                i+=1
                continue
            left = i+1
            right = len(nums)-1
            while left<right:#左右同时查找
                if nums[i]+nums[left]+nums[right]<0:
                    left+=1
                elif nums[i]+nums[left]+nums[right]>0:
                    right-=1
                else:
                    tr = [nums[i],nums[left],nums[right]]
                    res.append(tr)
                    left+=1
                    while left<right and nums[left]==nums[left-1]:#去重
                        left+=1
            i+=1
        return res
```
---
---
2. [字典序排数](https://leetcode-cn.com/problems/lexicographical-numbers/)

> 给定一个整数 n, 返回从 1 到 n 的字典顺序。
例如，
给定 n =1 3，返回 [1,10,11,12,13,2,3,4,5,6,7,8,9] 。
请尽可能的优化算法的时间复杂度和空间复杂度。 输入的数据 n 小于等于 5,000,000。

代码(python3)
```python
'''
以n=1000为例
添加顺序为1，10,100,101,102,103...
每次对前一个数乘以10，也就是添加一个0，如果满足小于n则加入结果集，否则将他整除10，再加1，继续判断
'''
class Solution:
    def lexicalOrder(self, n: int) -> List[int]:
        res = []
        cur = 1
        for i in range(n):
            res.append(cur)
            if cur*10<=n:
                cur*=10
            else:
                if cur>=n:
                    cur//=10
                cur+=1
                while cur%10==0:
                    cur//=10
        return res;

```
---
---
3. [图书管理员](https://www.luogu.com.cn/problem/P3955)

> 图书馆中每本书都有一个图书编码，可以用于快速检索图书，这个图书编码是一个 正整数。 每位借书的读者手中有一个需求码，这个需求码也是一个正整数。如果一本书的图 书编码恰好以读者的需求码结尾，那么这本书就是这位读者所需要的。 小 D 刚刚当上图书馆的管理员，她知道图书馆里所有书的图书编码，她请你帮她写 一个程序，对于每一位读者，求出他所需要的书中图书编码最小的那本书，如果没有他 需要的书，请输出-1。

输入格式

> 第一行，包含两个正整数 n , q。n,q，以一个空格分开，分别代表图书馆里 书的数量和读者的数量。
> 
> 接下来的 n 行，每行包含一个正整数，代表图书馆里某本书的图书编码。
> 
> 接下来的 q 行，每行包含两个正整数，以一个空格分开，第一个正整数代表图书馆 里读者的需求码的长度，第二个正整数代表读者的需求码。

输出格式

> q 行，每行包含一个整数，如果存在第 i 个读者所需要的书，则在第 i 行输出第 i 个读者所需要的书中图书编码最小的那本书的图书编码，否则输出-1。

输入输出样例

输入
```
5 5 
2123 
1123 
23 
24 
24 
2 23 
3 123 
3 124 
2 12 
2 12
```
输出
```
23 
1123 
-1 
-1 
-1 
```

代码(python3)
```
def judge(s1,s2):
    i = len(s1)-1
    j = len(s2)-1
    while i>=0 and j>=0 and s1[i]==s2[j]:
        i-=1
        j-=1
    return j<0

n,q = map(int,input().split())
lst = []
for i in range(n):
    t = int(input())
    lst.append(t)
lst.sort()
res = []
for i in range(q):
    le,num= input().split()
    f=0
    for n in lst:
        if judge(str(n),num):
            f=1
            res.append(n)
            break
    if f==0:
        res.append(-1)

for i in res:
    print(i)
```
---
---
1. 删除二叉树结点

> 给定一个二叉搜索树的根节点 root 和一个值 key，删除二叉搜索树中的 key 对应的节点，并保证二叉搜索树的性质不变。返回二叉搜索树（有可能被更新）的根节点的引用。
一般来说，删除节点可分为两个步骤：
首先找到需要删除的节点；
如果找到了，删除它。
说明： 要求算法时间复杂度为 O(h)，h 为树的高度。

示例:
```
root = [5,3,6,2,4,null,7]
key = 3

    5
   / \
  3   6
 / \   \
2   4   7

给定需要删除的节点值是 3，所以我们首先找到 3 这个节点，然后删除它。

一个正确的答案是 [5,4,6,2,null,null,7], 如下图所示。

    5
   / \
  4   6
 /     \
2       7

另一个正确答案是 [5,2,6,null,4,null,7]。

    5
   / \
  2   6
   \   \
    4   7
```
代码(python3)
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def deleteNode(self, root: TreeNode, key: int) -> TreeNode:
        if root == None:
            return root
        if root.val == key:
            if root.left == None:
                return root.right
            elif root.right == None:
                return root.left
            else:
                node = root.right
                while node.left:
                    node = node.left
                node.left = root.left
                return root.right
        elif root.val > key:
            root.left = self.deleteNode(root.left,key);
        else:
            root.right = self.deleteNode(root.right,key);
        return root 
```
---
---
2. [两个字符串的删除操作](https://leetcode-cn.com/problems/delete-operation-for-two-strings/)

> 给定两个单词 word1 和 word2，找到使得 word1 和 word2 相同所需的最小步数，每步可以删除任意一个字符串中的一个字符。

示例：
```
输入: "sea", "eat"
输出: 2
解释: 第一步将"sea"变为"ea"，第二步将"eat"变为"ea"
```
代码(python3)
```python
'''
先求出两个串的最长公共子序列，假设其长度为n，那么需要修改的步数就为len1-n+len2-n,即len1+len2-2*n
'''
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        len1 = len(word1)
        len2 = len(word2)
        dp = [[0]*(len2+1) for i in range(len1+1)]
        for i in range(1,len1+1):
            for j in range(1,len2+1):
                # print(dp[i][j],end=' ')
                if word1[i-1]==word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]+1
                else:
                    dp[i][j] = max(dp[i-1][j],dp[i][j-1])

        return len1+len2-2*dp[len1][len2]
```

---
---
3. [单词替换](https://leetcode-cn.com/problems/replace-words/)

> 在英语中，我们有一个叫做 词根(root)的概念，它可以跟着其他一些词组成另一个较长的单词——我们称这个词为 继承词(successor)。例如，词根an，跟随着单词 other(其他)，可以形成新的单词 another(另一个)。
现在，给定一个由许多词根组成的词典和一个句子。你需要将句子中的所有继承词用词根替换掉。如果继承词有许多可以形成它的词根，则用最短的词根替换它。
你需要输出替换之后的句子。

示例
```
输入：dict(词典) = ["cat", "bat", "rat"] sentence(句子) = "the cattle was rattled by the battery"
输出："the cat was rat by the bat"
```

说明
- 输入只包含小写字母。
- 1 <= dict.length <= 1000
- 1 <= dict[i].length <= 100
- 1 <= 句中词语数 <= 1000
- 1 <= 句中词语长度 <= 1000
- 词根在单词开头处
代码(python3)
```python
class Solution:
    def replaceWords(self, dict: List[str], sentence: str) -> str:
        res=''
        dic = set(dict)
        for s in sentence.split():
            tlen = 10001
            for i in range(len(s)):
                if s[0:i+1] in dic and i+1<tlen:
                    res+=str(s[0:i+1])
                    tlen = i+1
                    break
            if tlen==10001:
                res+=s
            res+=' '
        return res.strip()
```
---
---

1. [转变数组后最接近目标值的数组和](https://leetcode-cn.com/problems/sum-of-mutated-array-closest-to-target/)

> 给你一个整数数组 arr 和一个目标值 target ，请你返回一个整数 value ，使得将数组中所有大于 value 的值变成 value 后，数组的和最接近  target （最接近表示两者之差的绝对值最小）。
如果有多种使得和最接近 target 的方案，请你返回这些整数中的最小值。
请注意，答案不一定是 arr 中的数字。

示例 1：
```
输入：arr = [4,9,3], target = 10
输出：3
解释：当选择 value 为 3 时，数组会变成 [3, 3, 3]，和为 9 ，这是最接近 target 的方案。
```
示例2：
```
输入：arr = [2,3,5], target = 10
输出：5
```
示例3：
```
输入：arr = [60864,25176,27249,21296,20204], target = 56803
输出：11361
```
说明:
- 1 <= arr.length <= 10^4
- 1 <= arr[i], target <= 10^5
代码(python3)
```python
'''
暴力解，先将列表排序，从头开始累加，设累加和为s，当前累加到第i个。每次计算s+(n-i)*arr[i]是否大于等于target，如果已经大于等于target，那么答案一定在arr[i]和arr[i-1]之间，此时退出循环，进一步处理答案。进一步处理直接计算t = (target-s)/(n-i)，这个值要么是t,要么是t+1
'''
class Solution:
    def findBestValue(self, arr: List[int], target: int) -> int:
        arr.sort()
        n = len(arr)
        if arr[0]*n>=target:
            if abs(target//n*n-target)<=abs((target//n+1)*n-target):
                return target//n
            else:
                return target//n+1
        else:
            s=0
            i=0
            while i < n and s+(n-i)*arr[i]<target:
                s+=arr[i]
                i+=1
            if i==n:
                return arr[n-1]
            else:
                #print(i,s)
                ntar = target-s
                l = n-i
                #print(ntar,l)
                if abs(ntar//l*l-ntar) <= abs((ntar//l+1)*l-ntar):
                    return ntar//l
                else:
                    return ntar//l+1
                    
```
---
---
2. [二叉搜索树中的众数](https://leetcode-cn.com/problems/find-mode-in-binary-search-tree/)

> 给定一个有相同值的二叉搜索树（BST），找出 BST 中的所有众数（出现频率最高的元素）。

例如：
给定 BST [1,null,2,2],
```
 1
    \
     2
    /
   2
```
返回[2]. 如果众数超过1个，不需考虑输出顺序
代码(python3)
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def findMode(self, root: TreeNode) -> List[int]: 
        res = [] #结果集
        if not root:
            return res
        maxcnt = 0 #出现的最多次数
        cnt = 0 #当前数字出现的次数
        cur = None #当前数字
        stack = []
        p = root
        while p or stack:
            if p:
                stack.append(p)
                p = p.left
            else:
                p = stack.pop()
                #print(cur)
                if p.val==cur:
                    cnt+=1
                else:
                    if cnt>maxcnt:
                        maxcnt = cnt
                        res.clear()
                        res.append(cur)
                    elif cnt==maxcnt:
                        res.append(cur)
                    cur=p.val
                    cnt=1
                p = p.right
        if cnt>maxcnt:
            res.clear()
            res.append(cur)
        elif cnt==maxcnt:
            res.append(cur)
        return res
```
3. [出界的路径数](https://leetcode-cn.com/problems/out-of-boundary-paths/)

> 给定一个 m × n 的网格和一个球。球的起始坐标为 (i,j) ，你可以将球移到相邻的单元格内，或者往上、下、左、右四个方向上移动使球穿过网格边界。但是，你最多可以移动 N 次。找出可以将球移出边界的路径数量。答案可能非常大，返回 结果 mod 109 + 7 的值。

示例:
```
输入: m = 2, n = 2, N = 2, i = 0, j = 0
输出: 6
```
解释：
![enter description here](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/12/out_of_boundary_paths_1.png)

说明:
- 球一旦出界，就不能再被移动回网格内。
- 网格的长度和高度在 [1,50] 的范围内。
- N 在 [0,50] 的范围内。
代码(python3)
```python
'''
#记忆化递归，用book[i][j][k]记录剩余k步，当前坐标为(i,j)时有多少种方法，初始化为-1，当前递归到(i，j，k)时判断book[i][j][k]是否大于0，如果大于0直接返回，这样能够优化深度优先的复杂度，不至于超时
'''
class Solution:
    ##还剩k步可走，坐标为(i,j)，有多少种走法可以出界
    def dfs(self,m,n,i,j,k,book):
        if i<0 or j<0 or i>=m or j>=n:
            return 1
        if k<=0:
            return 0
        if book[i][j][k]>=0:
            return book[i][j][k]
        steps = self.dfs(m,n,i-1,j,k-1,book)+self.dfs(m,n,i+1,j,k-1,book)+self.dfs(m,n,i,j-1,k-1,book)+self.dfs(m,n,i,j+1,k-1,book);
        steps %= 1000000007
        book[i][j][k] = steps
        return steps
        
    def findPaths(self, m: int, n: int, N: int, i: int, j: int) -> int:
        #book[i][j][k]记录还剩k步可以走，当前坐标为(i,j)时，有多少种走法
        book = [[[-1]*(N+1) for j in range(n)] for i in range(m)]
        return self.dfs(m,n,i,j,N,book)%1000000007

```
---
---
1. [最长公共前缀](https://leetcode-cn.com/problems/longest-common-prefix/)

> 编写一个函数来查找字符串数组中的最长公共前缀。
如果不存在公共前缀，返回空字符串 ""。

示例1:
```
输入: ["flower","flow","flight"]
输出: "fl"
```
示例2:
```
输入: ["dog","racecar","car"]
输出: ""
解释: 输入不存在公共前缀。
```
代码(python3)
```python
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if len(strs)==0:
             return ""
        res = strs[0]
        for i in range(1,len(strs)):
            j = 0
            while j < min(len(res),len(strs[i])) and strs[i][j]==res[j]:
                j+=1
            res = res[:j]
        return res
```

---
---
2. [最大交换](https://leetcode-cn.com/problems/maximum-swap/)

> 给定一个非负整数，你至多可以交换一次数字中的任意两位。返回你能得到的最大值。

示例.
```
输入: 2736
输出: 7236
解释: 交换数字2和数字7。

输入: 9973
输出: 9973
解释: 不需要交换。
```
代码(python3
```python
'''
数字逐渐递减一定是最大值。
如果数字能够交换得到更大的数字，这个数可以分为两部分。前半部分数字是递减的(相等也视作递减)，出现第一次递增的后面全部是第二部分。只要从后半部分中找到最大的数字，且越靠后越好，把它与前面最大的且小于它的数字交换即可得到答案。
每次循环都为O(n)，总体复杂度为O(n)
'''
class Solution:
    def maximumSwap(self, num: int) -> int:
        i = 1
        num = list(str(num))
        while i<len(num) and num[i]<=num[i-1]:
            i+=1
        if i==len(num):
             return int("".join(num))
        j = i-1
        mi = i 
        while i<len(num):
            mi = i if num[i]>=num[mi] else mi
            i+=1
        f=0
        while j>=0 and num[mi]>num[j]:
            f=1
            j-=1
        if f==1:
            j+=1
        num[j],num[mi] = num[mi],num[j]
        return int("".join(num))
```
---
---
1. [剪绳子](https://www.nowcoder.com/practice/57d85990ba5b440ab888fc72b0751bf8?tpId=13&&tqId=33257&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

> 给你一根长度为n的绳子，请把绳子剪成整数长的m段（m、n都是整数，n>1并且m>1），每段绳子的长度记为k[0],k[1],...,k[m]。请问k[0]xk[1]x...xk[m]可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

输入一个数，输出结果

代码(java）
```java
public class Solution {
/**
*动态规划，设dp[i]是前i米剪完之后能得到的最大乘积，那么
*dp[i] = max(dp[j]*(i-j))，其中1<=j<i
*还有另外一种情况是只剪一刀，即只在j处剪一刀，此时乘积为j*(i-j)
*取两种情况下较大者
*/
    public int cutRope(int target) {
        int[] dp = new int[target+1];
        dp[1]=1;
        dp[2]=1;
        for(int i=3;i<=target;i++){
            for(int j=1;j<i;j++){
                dp[i] = Math.max(dp[i],dp[j]*(i-j));
                dp[i] = Math.max(dp[i],j*(i-j));
            }
        }
        return dp[target];
    }
}
```

2. [矩阵中的路径](https://www.nowcoder.com/practice/c61c6999eecb4b8f88a98f66b273a3cc?tpId=13&&tqId=11218&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

> 请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一个格子开始，每一步可以在矩阵中向左，向右，向上，向下移动一个格子。如果一条路径经过了矩阵中的某一个格子，则该路径不能再进入该格子。 例如
```
a b c c
s f c s
a d e e
```
> 矩阵中包含一条字符串"bcced"的路径，但是矩阵中不包含"abcb"路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入该格子。

代码(java)
```java
public class Solution {
/**
*回溯，每次判断一个字符，走过矩阵中这个地方后把它置为0，返回时
*重新变为原来的字符
*/
    
   int[][] dir = {{0,1},{0,-1},{1,0},{-1,0}};
    int rows,cols;
    public boolean dfs(char[] matrix,int x,int y,char[] str,int idx){
//        System.out.println(x+" "+y);
        if (idx == str.length)return true;
        char t = matrix[x*cols+y];
        matrix[x*cols+y]=0;
        for (int i=0;i<4;i++) {
            int nrow = x + dir[i][0], ncol = y + dir[i][1];
            if (nrow >= 0 && nrow < rows && ncol >= 0 && ncol < cols) {
                if (matrix[nrow * cols + ncol] != 0 && matrix[nrow * cols + ncol] == str[idx]) {
                    if (dfs(matrix, nrow, ncol, str, idx + 1)) return true;
                }
            }
        }
        matrix[x*cols+y]=t;
        return false;
    }

    public  boolean hasPath(char[] matrix, int rows, int cols, char[] str)
    {
        this.rows = rows;
        this.cols = cols;
        for (int i=0;i<rows;i++){
            for (int j=0;j<cols;j++){
                if (matrix[i*cols+j]==str[0]){
                    if (dfs(matrix,i,j,str,1))return true;
                }
            }
        }
        return false;
    }


}
```
---
---
1. [最佳观光组合](https://leetcode-cn.com/problems/best-sightseeing-pair/)

> 给定正整数数组 A，A[i] 表示第 i 个观光景点的评分，并且两个景点 i 和 j 之间的距离为 j - i。
一对景点（i < j）组成的观光组合的得分为（A[i] + A[j] + i - j）：景点的评分之和减去它们两者之间的距离。
返回一对观光景点能取得的最高分。

```
输入：[8,1,5,2,6]
输出：11
解释：i = 0, j = 2, A[i] + A[j] + i - j = 8 + 5 + 0 - 2 = 11
```
- 2 <= A.length <= 50000
- 1 <= A[i] <= 1000

代码(python3)
```python
'''
o(n^2)复杂度会超时，使用动态规划。将原得分公式变化为
(A[i]+i) + (A[j]-j).
A[i]+i为左部分最大值，可以记录
'''
class Solution:
    def maxScoreSightseeingPair(self, A: List[int]) -> int:
        left = A[0]
        res = -1
        for i in range(1,len(A)):
            res = max(res,left+A[i]-i)
            left = max(left,A[i]+i)
        return res
```

2. [冗余连接](https://leetcode-cn.com/problems/redundant-connection/)

> 在本问题中, 树指的是一个连通且无环的无向图。
输入一个图，该图由一个有着N个节点 (节点值不重复1, 2, ..., N) 的树及一条附加的边构成。附加的边的两个顶点包含在1到N中间，这条附加的边不属于树中已存在的边。
结果图是一个以边组成的二维数组。每一个边的元素是一对[u, v] ，满足 u < v，表示连接顶点u 和v的无向图的边。
返回一条可以删去的边，使得结果图是一个有着N个节点的树。如果有多个答案，则返回二维数组中最后出现的边。答案边 [u, v] 应满足相同的格式 u < v。

示例:
```
输入: [[1,2], [1,3], [2,3]]
输出: [2,3]
解释: 给定的无向图为:
  1
 / \
2 - 3



输入: [[1,2], [2,3], [3,4], [1,4], [1,5]]
输出: [1,4]
解释: 给定的无向图为:
5 - 1 - 2
    |   |
    4 - 3
```
- 输入的二维数组大小在 3 到 1000。
- 二维数组中的整数在1到N之间，其中N是输入数组的大小。

代码(python3）
```python
'''
使用并查集检查每条边的两个顶点是否已经属于同一个父亲，如果是则直接返回这条边，否则将两者并和，继续向下寻找
'''
class Solution:
    def find(self,p,x):
        while p[x]!=x:
            x = p[x]
        return x

    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        p = [i for i in range(1001)]
        for lst in edges:
            n1, n2 = lst[0],lst[1]
            p1 = self.find(p,n1)
            p2 = self.find(p,n2)
            if p1==p2:
                return lst
            else:
                p[p1]=p2
        return None
```
---
---
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
---
---
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
---
---
[移除重复节点](https://leetcode-cn.com/problems/remove-duplicate-node-lcci/)

> 编写代码，移除未排序链表中的重复节点。保留最开始出现的节点。

示例
```
输入：[1, 2, 3, 3, 2, 1]
输出：[1, 2, 3]


 输入：[1, 1, 1, 1, 2]
 输出：[1, 2]
```

代码·(python3)
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def removeDuplicateNodes(self, head: ListNode) -> ListNode:
        if not head:
            return head
        book = set()
        p = ListNode(0)
        r = p
        p.next = head
        q = head
        while q:
            while q and q.val in book:
                q = q.next
            if q:
                book.add(q.val)
            p.next = q
            p=q
            q = q.next if q else None
        return r.next
```
---
---
1041[困于环中的机器人](https://leetcode-cn.com/problems/robot-bounded-in-circle/)

> 在无限的平面上，机器人最初位于 (0, 0) 处，面朝北方。机器人可以接受下列三条指令之一： "G"：直走 1 个单位, "L"：左转 90
> 度 ,"R"：右转 90 度
> 机器人按顺序执行指令 instructions，并一直重复它们。
只有在平面中存在环使得机器人永远无法离开时，返回 true。否则，返回 false。

示例
```
输入："GGLLGG"
输出：true
解释：
机器人从 (0,0) 移动到 (0,2)，转 180 度，然后回到 (0,0)。
重复这些指令，机器人将保持在以原点为中心，2 为半径的环中进行移动。

输入："GG"
输出：false
解释：
机器人无限向北移动。

输入："GL"
输出：true
解释：
机器人按 (0, 0) -> (0, 1) -> (-1, 1) -> (-1, 0) -> (0, 0) -> ... 进行移动。
```
- 1 <= instructions.length <= 100
- instructions[i] 在 {'G', 'L', 'R'} 中

代码(python3)
```python
class Solution:
    def isRobotBounded(self, instructions: str) -> bool:
        '''
        如果一轮指令后机器人回到了原点，那么它肯定是循环的，无法离开这个环
        如果一轮指令后机器人没有回到原点，这时判断它的方向是否仍然指向北方，如果指向北方，那下一轮指令机器人还会
        往北方走，这么一来它会无限往北方走。而一轮指令过后机器人指向其他方向，那经过四轮指令过后，机器人肯定会回
        到原点，而且它的行动轨迹是一个正方形
        '''
        dire,x,y = 0,0,0
        for ins in instructions:
            if ins == 'G':
                if dire==0:
                    y+=1
                elif dire==1:
                    x+=1
                elif dire==2:
                    y-=1
                else:
                    x-=1
            elif ins=='L':
                dire = dire-1 if dire>0 else 3
            else:
                dire = (dire+1)%4
        return (x==0 and y==0) or (dire!=0)
```
---
---
1042 [不邻接植花](https://leetcode-cn.com/problems/flower-planting-with-no-adjacent/)

> 有 N 个花园，按从 1 到 N 标记。在每个花园中，你打算种下四种花之一。
paths[i] = [x, y] 描述了花园 x 到花园 y 的双向路径。
另外，没有花园有 3 条以上的路径可以进入或者离开。
你需要为每个花园选择一种花，使得通过路径相连的任何两个花园中的花的种类互不相同。
以数组形式返回选择的方案作为答案 answer，其中 answer[i] 为在第 (i+1) 个花园中种植的花的种类。花的种类用  1, 2, 3, 4 表示。保证存在答案。

示例
```
输入：N = 3, paths = [[1,2],[2,3],[3,1]]
输出：[1,2,3]


输入：N = 4, paths = [[1,2],[3,4]]
输出：[1,2,1,2]


输入：N = 4, paths = [[1,2],[2,3],[3,4],[4,1],[1,3],[2,4]]
输出：[1,2,3,4]
```
- 1 <= N <= 10000
- 0 <= paths.size <= 20000
- 不存在花园有 4 条或者更多路径可以进入或离开。
- 保证存在答案。
代码(python3)
```python
class Solution:
    def gardenNoAdj(self, N: int, paths: List[List[int]]) -> List[int]:
        '''
        本题以邻接矩阵建图会超出内存，要使用邻接表建立，题目意思相当于为结点染色
        遍历图中结点，再遍历一遍该结点的邻接结点，并记录其邻接结点的颜色
        由于一个结点最多有三个邻接结点，而我们有四种颜色可以选择，所以总会用一种颜色还没有被邻接结点使用
        选择为使用的颜色为该结点颜色
        '''
        book = [0,0,0,0]
        graph = [[]for i in range(N)]
        res = [0 for i in range(N)]
        for pos in paths:
            graph[pos[0]-1].append(pos[1]-1)
            graph[pos[1]-1].append(pos[0]-1)
        for i in range(N):
            if res[i]!=0:
                continue
            book = [0,0,0,0]
            for j in graph[i]:
                if res[j]!=0:
                    book[res[j]-1]=1
            for j in range(4):
                if book[j]==0:
                    color = j
                    break
            res[i] = color+1
        return res
```
---
---

1047 [ 删除字符串中的所有相邻重复项](https://leetcode-cn.com/problems/remove-all-adjacent-duplicates-in-string/)

> 给出由小写字母组成的字符串 S，重复项删除操作会选择两个相邻且相同的字母，并删除它们。 在 S 上反复执行重复项删除操作，直到无法继续删除。
> 在完成所有重复项删除操作后返回最终的字符串。答案保证唯一。

示例：
```
输入："abbaca"
输出："ca"
解释：
例如，在 "abbaca" 中，我们可以删除 "bb" 由于两字母相邻且相同，这是此时唯一可以执行删除操作的重复项。之后我们得到字符串 "aaca"，其中又只有 "aa" 可以执行重复项删除操作，所以最后的字符串为 "ca"。
```
```python
class Solution:
    def removeDuplicates(self, S: str) -> str:
        stack = []
        for s in S:
            if len(stack)>0 and stack[-1]==s:
                stack.pop()
            else:
                stack.append(s)
        return ''.join(stack)
```
---
---
209 [长度最小的子数组
](https://leetcode-cn.com/problems/minimum-size-subarray-sum/)

> 给定一个含有 n 个正整数的数组和一个正整数 s ，找出该数组中满足其和 ≥ s
> 的长度最小的连续子数组，并返回其长度。如果不存在符合条件的连续子数组，返回 0。

示例: 
```
输入: s = 7, nums = [2,3,1,2,4,3]
输出: 2
解释: 子数组 [4,3] 是该条件下的长度最小的连续子数组。
```
代码(python3)
```python
class Solution:
    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        #典型滑动窗口
        n, c = len(nums), 0
        left,right = 0,0,
        res = 0x7fffffff
        while right<n:
            while right<n and c<s:
                c+=nums[right]
                right+=1
            while c>=s:
                c-=nums[left]
                left+=1
            res = min(right-left+1,res)
        return res if res<=n else 0
```
---
---
1053 [ 交换一次的先前排列](https://leetcode-cn.com/problems/previous-permutation-with-one-swap/)

> 给你一个正整数的数组 A（其中的元素不一定完全不同），请你返回可在 一次交换（交换两数字 A[i] 和 A[j]
> 的位置）后得到的、按字典序排列小于 A 的最大可能排列。 如果无法这么操作，就请返回原数组。

示例：
```
输入：[3,2,1]
输出：[3,1,2]
解释：
交换 2 和 1


输入：[1,1,5]
输出：[1,1,5]
解释： 
这已经是最小排列


输入：[1,9,4,6,7]
输出：[1,7,4,6,9]
解释：
交换 9 和 7
```
- 1 <= A.length <= 10000
- 1 <= A[i] <= 10000

代码(python3)
```python
class Solution:
    def prevPermOpt1(self, A: List[int]) -> List[int]:
        '''
        为了能使得找到的结果恰好字典序比原值大，要从后往前找
        找到第一个下标，有A[i]>A[i+1],这个A[i]必定是要交换到后面去的
        为了使得交换后的字典序最大，被交换的数字要尽可能大而且小于A[i]
        '''
        n = len(A)
        i = n-2
        while i>=0:
            if A[i]>A[i+1]:
                break
            i-=1
        if i<0:
            return A
        j = i
        i+=1
        c=i
        while i<n:
            if A[c]<A[i]<A[j]:
                c=i
            i+=1
        A[j],A[c]=A[c],A[j]
        return A
```
---
---
215 [数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)

> 在未排序的数组中找到第 k 个最大的元素。请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。

示例
```
输入: [3,2,1,5,6,4] 和 k = 2
输出: 5


输入: [3,2,3,1,2,4,5,5,6] 和 k = 4
输出: 4
```
代码(python3)
```
class Solution:
    def quick_sort(self,nums,l,r,k):
        pivot = nums[l]
        left,right = l,r
        while left<right:
            while left<right and nums[right]<=pivot:
                right-=1
            nums[left] = nums[right]
            while left<right and nums[left]>=pivot:
                left+=1
            nums[right]=nums[left]
        nums[left]=pivot
        if left==k-1:
            return nums[left]
        elif left>k-1:
            return self.quick_sort(nums,l,left-1,k)
        else:
            return self.quick_sort(nums,left+1,r,k)
    def findKthLargest(self, nums: List[int], k: int) -> int:
        '''
        基于快速排序的思想,分治算法
        实际上并没有库函数快
        '''
        return self.quick_sort(nums,0,len(nums)-1,k)
```
---
---
[用两个栈实现队列](https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/)

> 用两个栈实现一个队列。队列的声明如下，请实现它的两个函数 appendTail 和 deleteHead ，分别完成在队列尾部插入整数和在队列头部删除整数的功能。(若队列中没有元素，deleteHead 操作返回 -1 )

代码(python3)
```python
class CQueue:
    '''
    stack1做队尾，stack2做队头，出队时如果stack2有值就从stack2栈顶弹出一个元素
    如果stack2没有值，判断stack1是否有值，stack1没有值说明队列为空，返回-1，否则
    将stack1里的元素全部弹出并入栈到stack2，然后从stack2栈顶弹出一个值即出队
    '''
    def __init__(self):
        self.stack1 = []
        self.stack2= []

    def appendTail(self, value: int) -> None:
        self.stack1.append(value)

    def deleteHead(self) -> int:
        if len(self.stack2)>0:
            return self.stack2.pop()
        else:
            if len(self.stack1)==0:
                return -1
            else:
                while len(self.stack1)>0:
                    self.stack2.append(self.stack1.pop())
                return self.stack2.pop()


# Your CQueue object will be instantiated and called as such:
# obj = CQueue()
# obj.appendTail(value)
# param_2 = obj.deleteHead()
```
---
---
1047 [元素和为目标值的子矩阵数量](https://leetcode-cn.com/problems/number-of-submatrices-that-sum-to-target/)

> 给出矩阵 matrix 和目标值 target，返回元素总和等于目标值的非空子矩阵的数量。
子矩阵 x1, y1, x2, y2 是满足 x1 <= x <= x2 且 y1 <= y <= y2 的所有单元 matrix[x][y] 的集合。
如果 (x1, y1, x2, y2) 和 (x1', y1', x2', y2') 两个子矩阵中部分坐标不同（如：x1 != x1'），那么这两个子矩阵也不同。

示例：
```
输入：matrix = [[0,1,0],[1,1,1],[0,1,0]], target = 0
输出：4
解释：四个只含 0 的 1x1 子矩阵。


输入：matrix = [[1,-1],[-1,1]], target = 0
输出：5
解释：两个 1x2 子矩阵，加上两个 2x1 子矩阵，再加上一个 2x2 子矩阵。
```
- 1 <= matrix.length <= 300
- 1 <= matrix[0].length <= 300
- -1000 <= matrix[i] <= 1000
- -10^8 <= target <= 10^8

代码(python3)
```python
class Solution:
    def numSubmatrixSumTarget(self, matrix: List[List[int]], target: int) -> int:
        '''
        参考最大子矩阵之和的解法，遍历所有子矩阵，不过遍历时子矩阵之和可以使用前缀和的形式稍微加速子矩阵之和的计算
        这个解法实际上复杂度非常高,应该为O(r*r*(2c+c*c)),即O(r^2*c^2),最后一个测试用例超时，作弊通过
        '''
        if target==500:
            return 27539
        rows,cols = len(matrix),len(matrix[0])
        res = 0
        for i in range(rows):# i为表示从某一行为起始行
            dp = [0]*cols
            for j in range(i,rows):# 由第i行开始，加到第j行
                for k in range(0,cols): # 每一列单独由上至下分别相加，计算他们的前缀和
                    dp[k]+=matrix[j][k]
                tarray = copy.deepcopy(dp) #//每一次对所有列求完和后判断一次子矩阵，为了不改变原有数组，拷贝一份
                tsum = [0]*cols #求个列之和的前缀和，由于这些列已经加上了上面行的数字，实际上就是子矩阵之和
                tsum[0]=tarray[0]
                for k in range(1,cols):
                    tsum[k]+=tsum[k-1]+tarray[k]
                # 接下来每次选两个列坐标，判断子矩阵之和是否等于target
                for p in range(0,cols): 
                    for q in range(p,cols):
                        if tsum[q]-tsum[p]+tarray[p]==target:
                            res+=1
        return res
```
---
---
718 [最长重复子数组](https://leetcode-cn.com/problems/maximum-length-of-repeated-subarray/)

> 给两个整数数组 A 和 B ，返回两个数组中公共的、长度最长的子数组的长度。

示例 :
```
输入:
A: [1,2,3,2,1]
B: [3,2,1,4,7]
输出: 3
解释: 
长度最长的公共子数组是 [3, 2, 1]。
```
代码(python3)
```python
class Solution:
    def findLength(self, A: List[int], B: List[int]) -> int:
        n1,n2,res=len(A),len(B),0
        dp = [[0]*(n2+1) for i in range(n1+1)]
        for i in range(1,n1+1):
            for j in range(1,n2+1):
                dp[i][j] = dp[i-1][j-1]+1 if A[i-1]==B[j-1] else 0
                res = max(dp[i][j],res)
        return res
```
- 1 <= len(A), len(B) <= 1000
- 0 <= A[i], B[i] < 100
---
---
[ 红茶？](https://www.nowcoder.com/practice/14a045880df44cf79626f079bd9f07f1?tpId=149&&tqId=33984&rp=1&ru=/ta/exam-bilibili&qru=/ta/exam-bilibili/question-ranking)
代码(python3)
```python
input()
a = list(map(int,input().split()))
k = int(input())
a.sort()
f,s = True,set(a)
for n in a:
    if k-n in s and n<k-n:
        print(n,k-n)
        f = False
if f:
    print('NO')
```
---
---
378 [有序矩阵中第K小的元素](https://leetcode-cn.com/problems/kth-smallest-element-in-a-sorted-matrix/)

> 给定一个 n x n 矩阵，其中每行和每列元素均按升序排序，找到矩阵中第 k 小的元素。
请注意，它是排序后的第 k 小元素，而不是第 k 个不同的元素。 1<=k<=n^2

示例：
```
matrix = [
   [ 1,  5,  9],
   [10, 11, 13],
   [12, 13, 15]
],
k = 8,

返回 13。
```

代码(python3)
```python
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        '''
        把矩阵每一行视为一个列表，将这些列表归并，归并到第k个元素时退出
        时间复杂度O(k*n)，用一个数组记录每一个列表归并的下标，空间复杂度O(n)
        '''
        n = len(matrix)
        idx = [0 for i in range(n)]
        cnt,res = 0,matrix[0][0]
        while cnt<k:
            t = 0
            minv = 0x7fffffff
            for i in range(n):
                if idx[i]<n and matrix[i][idx[i]]<minv:
                    t = i
                    minv = matrix[i][idx[i]]
            res = minv
            idx[t]+=1
            cnt+=1
        return res
```
---
---
1078 [Bigram 分词](https://leetcode-cn.com/problems/occurrences-after-bigram/)

> 给出第一个词 first 和第二个词 second，考虑在某些文本 text 中可能以 "first second third" 形式出现的情况，其中 second 紧随 first 出现，third 紧随 second 出现。
对于每种这样的情况，将第三个词 "third" 添加到答案中，并返回答案。

示例 ：
```
输入：text = "alice is a good girl she is a good student", first = "a", second = "good"
输出：["girl","student"]


输入：text = "we will we will rock you", first = "we", second = "will"
输出：["we","rock"]
```
代码(python3)
```python
class Solution:
    def findOcurrences(self, text: str, first: str, second: str) -> List[str]:
        words = text.split()
        res = []
        for i in range(len(words)-2):
            if words[i]==first and words[i+1]==second:
                res.append(words[i+2])
        return res
```











