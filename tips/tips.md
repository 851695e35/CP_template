LL 耗时较大 -》 double

comb , qpow 常数较大，如果数据范围较小建议先预处理

官方 非mod的 pow() 是 double 不好用，自己写一个很快。





binary_search:

排序好的数组，搜索是否包含一个元素，return true/false

```c++c++
binary_search(a.begin(), a.end(), 15)
```



如果需要用mp找最大值和第二大，

用 `auto it = prev(mp.end())` instead of `auto it = mp.rbegin()` , 后者会在 prev 的时候出问题。



如果直接输出 double 类精度有问题，需要 setprecision
