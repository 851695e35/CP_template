## segtree

维护区间最值为例：

```c++
struct T
{
    int l,r,mid;
    int lazy,sum;
}tree[N<<2];
void build(int rt,int l,int r)
{
    tree[rt].lazy=0;
    tree[rt].l=l;
    tree[rt].r=r;
    if(l==r)
    {
        tree[rt].sum=a[l];//依据初始化类型具体而定，置0可以看为空树，但是l,r均初始化。
        return ;
    }
    int mid=tree[rt].mid=l+r>>1;
    build(rt<<1,l,mid);
    build(rt<<1|1,mid+1,r);
}
void push_down(int rt)
{
    if(tree[rt].lazy)
    {
        tree[rt<<1].sum+=tree[rt].lazy;
        tree[rt<<1].lazy+=tree[rt].lazy;
        tree[rt<<1|1].sum+=tree[rt].lazy;
        tree[rt<<1|1].lazy+=tree[rt].lazy;
        tree[rt].lazy=0;
    }
}
void push_up(int rt){
    tree[rt].sum = max(tree[rt<<1].sum,tree[rt<<1|1].sum);//看情况而定
}
void update(int rt,int l,int r,int v)
{
    if(tree[rt].r<l||tree[rt].l>r) return ;
    if(tree[rt].l>=l&&tree[rt].r<=r)
    {
        tree[rt].sum+=v;
        tree[rt].lazy+=v;
        return ;
    }
    push_down(rt);
    if(tree[rt].mid>=l) update(rt<<1,l,r,v);
    if(tree[rt].mid<r) update(rt<<1|1,l,r,v);
    push_up(rt);
}
int query(int rt,int l,int r)
{
    if(tree[rt].r<l||tree[rt].l>r) return 0;
    if(tree[rt].l>=l&&tree[rt].r<=r) return tree[rt].sum;
    push_down(rt);
    if(tree[rt].mid<l) return query(rt<<1|1,l,r); 
    if(tree[rt].mid>=r) return query(rt<<1,l,r);
    return max(query(rt<<1|1,l,r), query(rt<<1,l,r));
}
```



## BIT (Fenwick)

template from jiangly 注意 add 当下， sum（x) = x-1的sum  有超时风险，因为vector

```C++
template <class T>
struct Fenwick {
    int n;
    vector<T> a;
    
    Fenwick(int n = 0) {
        this->n = n;
        a.assign(n, T());
    }
    
    void add(int x, T v) {
        for (int i = x + 1; i <= n; i += i & -i) {
            a[i - 1] += v;
        }
    }
    T sum(int x) {
        auto ans = T();
        for (int i = x; i > 0; i -= i & -i) {
            ans += a[i - 1];
        }
        return ans;
    }
    
    T rangeSum(int l, int r) {
        return sum(r) - sum(l);
    }
    
};
```



## exgcd

求解：ax+by = gcd(a,b)

```c++
ll exgcd(ll a,ll b,ll &x,ll &y)
{
	if(!b)
	{
		x=1;y=0;
		return a;
	}
	else
	{
		ll tx,ty;
		ll d=exgcd(b,a%b,tx,ty);
		x=ty;y=tx-(a/b)*ty;
		return d;
	}
}
```





## DSU

```c++
struct DSU {
    vector<int> f, siz;
    
    DSU() {}
    DSU(int n) {
        init(n);
    }
    
    void init(int n) {
        f.resize(n);
        iota(f.begin(), f.end(), 0);
        siz.assign(n, 1);
    }
    
    int find(int x) {
        while (x != f[x]) {
            x = f[x] = f[f[x]];
        }
        return x;
    }
    
    bool same(int x, int y) {
        return find(x) == find(y);
    }
    
    bool merge(int x, int y) {
        x = find(x);
        y = find(y);
        if (x == y) {
            return false;
        }
        siz[x] += siz[y];
        f[y] = x;
        return true;
    }
    
    int size(int x) {
        return siz[find(x)];
    }
};
```



## tarjan

使用tarjan算法求强连通分量。

有向图（对于无向图还需要加一个int fa(若需要求环)）

```c++
vector<int>adj[N];
int dfn[N],low[N],stk[N],top,dn,vis[N];
int nsz,nl[N];
void tarjan(int u)
{
    vis[u] = 1;
    dfn[u] = low[u] = ++dn;
    stk[++top] = u;
    for(auto v:adj[u]){
        if(!dfn[v]){
            tarjan(v);
            low[u] = min(low[u],low[v]);
        }
        else if(vis[v]){
            low[u] = min(low[u],dfn[v]);
        }
    }
    if(low[u]==dfn[u]){//强连通分量
        nl[u] = ++nsz;
        while(stk[top]!=u)nl[stk[top]] = nl[u],vis[stk[top]] = 0,top--;
        vis[u] = 0,top--;
    }
}
```

