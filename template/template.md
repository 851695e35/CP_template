### NTT

```C++
#define ll long long
const ll mod = 998244353;
const int N = (1<<22)+5;
ll qpow(ll a,ll b)
{
    ll ret = 1;
    while(b){
        if(b&1)ret =ret*a%mod;
        a = a*a%mod;
        b>>=1;
    }
    return ret;
}

using ull=unsigned long long;
using Poly=vector<ll>;
constexpr int g=3;
int maxn,f[20][N],rev[N],w[N];
void init(int len)
{
	int l=0;maxn=1;
	while(maxn<=len) maxn<<=1,l++;
	for(int i=0;i<maxn;i++) rev[i]=(rev[i>>1]>>1)|((i&1)<<(l-1));
	for(int i=1;i<maxn;i<<=1)
	{
		ll wm=qpow(g,(mod-1)/(i<<1));w[i]=1;
		for(int j=1;j<i;j++) w[i+j]=w[i+j-1]*wm%mod;
	}
}
void NTT(Poly &p,bool flag)
{
	static ull a[2100000];
	p.resize(maxn);
	for(int i=0;i<maxn;i++) a[i]=p[rev[i]];
	for(int i=1;i<maxn;i<<=1)
	{
		for(int j=0;j<maxn;j+=i<<1)
			for(int k=j;k<j+i;k++)
			{
				ull x=a[k],y=a[k+i]*w[i+k-j]%mod;
				a[k]=x+y;a[k+i]=x+mod-y;
			}
		if(i==2048) for(int j=0;j<maxn;j++) a[j]%=mod;
	}
	if(flag) for(int i=0;i<maxn;i++) p[i]=a[i]%mod;
	else
	{
		reverse(a+1,a+maxn);
		int inv=qpow(maxn,mod-2);
		for(int i=0;i<maxn;i++) p[i]=a[i]%mod*inv%mod;
	}
}
Poly mul(Poly a,Poly b,int n=-1)
{
	int sz1=a.size(),sz2=b.size();
	if(n<0) n=sz1+sz2-1;
	init(sz1+sz2);
	NTT(a,1);NTT(b,1);
	for(int i=0;i<maxn;i++) a[i]=(ll)a[i]*b[i]%mod;
	NTT(a,0);a.resize(n);
	return a;
}
```

分治ntt: 以求bigpi为例

```c++
void merge(int l,int r)
{
    ...
    int mid = (l+r)/2;
    merge(l,mid),merge(mid+1,r);
    //会有 nlogn 的卷积在这里 -》 总 T = (nlog^2n)
    ...
}
```

 

### SA

```c++
const int N =2E6+5;
int sa[N],c[N],h[N],a1[N],c1[N];
pair<char, int>t[N];
void suffix_array(string s){
    int n = s.size();
    for (int i = 0; i < n; i++) t[i]={s[i], i};
    sort(t, t+n);
    int cur = -1;
    for (int i = 0; i < n; i++){
        if (i == 0 || t[i].first != t[i - 1].first){
            cur++;
            h[cur] = i;
        }
        sa[i] = t[i].second;
        c[sa[i]] = cur;
    }
    for (int len = 1; len < n; len *= 2){
        for (int i = 0; i < n; i++){
            int j = (n + sa[i] - len) % n;
            a1[h[c[j]]++] = j;
        }
        for(int i = 0;i<n;i++)sa[i] = a1[i];
        cur = -1;
        for (int i = 0; i < n; i++){
            if (i == 0 || c[sa[i]] != c[sa[i - 1]] || c[(sa[i] + len) % n] != c[(sa[i - 1] + len) % n]){
                cur++;
                h[cur] = i;
            }
            c1[sa[i]] = cur;
        }
        for(int i = 0;i<n;i++)c[i] = c1[i];
    }
    return;
}
```



### FENWICK

```c++
struct Fenwick {
    int n;
    std::vector<T> a;
    
    Fenwick(int n = 0) {
        init(n);
    }
    
    void init(int n) {
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

```c++
struct Max {
    int v;
    Max(int x = -1E9) : v{x} {}
    
    Max &operator+=(Max a) {
        v = std::max(v, a.v);
        return *this;
    }
};
Fenwick<Max>fen(n)//or n+1 depends
```

直接用的版本：

```c++
int tr[N];
void add(int x,int n) {
    for (int i = x + 1; i <= n; i += i & -i) {
        tr[i - 1] += v;
    }
}
int sum(int x) { // (0,x-1]
    int ans = 0;
    for (int i = x; i > 0; i -= i & -i) {
        ans += tr[i - 1];
    }
    return ans;
}

int rangeSum(int l, int r) { //[l,r)
    return sum(r) - sum(l);
}
```



### SEGTREE

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



### DSU

```c++
int fa[N];

int find(int x) {
    if (x == fa[x]) return x;
    return fa[x] = find(fa[x]);
}

int merge(int x, int y) {
    int fx = find(x), fy = find(y);
    if (fx == fy) return 0;
    fa[fx] = fy;
    return 1;
}
```

### tarjan

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



### exgcd

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

### euler's sieve

```c++
void init(int n) {
  for (int i = 2; i <= n; ++i) {
    if (!vis[i]) {
      pri[cnt++] = i;
    }
    for (int j = 0; j < cnt; ++j) {
      if (1ll * i * pri[j] > n) break;
      vis[i * pri[j]] = 1;
      if (i % pri[j] == 0) {
        break;
      }
    }
  }
}
```

### Euler Phi

```c++
int phi(int n)
{
    int res = n;
    for (int i = 2; i * i <= n; i++)
    {
        if (n % i == 0)
            res = res / i * (i - 1); // 先除再乘防止溢出
        while (n % i == 0) // 每个质因数只处理一次，可以把已经找到的质因数除干净
            n /= i;
    }
    if (n > 1)
        res = res / n * (n - 1); // 最后剩下的部分也是原来的n的质因数
    return res;
}
```

### prime detection 1e12

#### miller-rabin

```c++
ll qmul(ll a,ll b,ll mod)//快速乘
{
    ll c=(long double)a/mod*b;
    ll res=(unsigned long long)a*b-(unsigned long long)c*mod;
    return (res+mod)%mod;
}
ll qpow(ll a,ll n,ll mod)//快速幂
{
    ll res=1;
    while(n)
    {
        if(n&1) res=qmul(res,a,mod);
        a=qmul(a,a,mod);
        n>>=1;
    }
    return res;
}
bool millerrabin(ll n)//Miller Rabin Test
{
    if(n<3||n%2==0) return n==2;//特判
    ll u=n-1,t=0;
    while(u%2==0) u/=2,++t;
    ll ud[]={2,325,9375,28178,450775,9780504,1795265022};
    for(ll a:ud)
    {
        ll v=qpow(a,u,n);
        if(v==1||v==n-1||v==0) continue;
        for(int j=1;j<=t;j++)
        {
            v=qmul(v,v,n);
            if(v==n-1&&j!=t){v=1;break;}//出现一个n-1，后面都是1，直接跳出
            if(v==1) return 0;//这里代表前面没有出现n-1这个解，二次检验失败
        }
        if(v!=1) return 0;//Fermat检验
    }
    return 1;
}
```



### 树哈希求同构

```c++
#include<chrono>
const ll mask = chrono::steady_clock::now().time_since_epoch().count();
ll shift(ll x) {
    x ^= mask;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x ^= mask;
    return x;
}
auto Treehash = [&](int root){
    function<ll(int,int)> dfs = [&](int u,int fa){
        ll h = 1;
        for(int v:g[u]){
            if(v==fa)continue;
            h+= shift(dfs(v,u));
        }
        return h;
    };
    return dfs(root,-1);
};
```

### n -> facbase

```c++
void get_fac(int x)
{
    for(int i = 2;i*i<=x;i++){
        if(x%i==0)
            while(x%i==0){
                mp[i]++;
                x/=i;
            }
    }
    if(x!=1)mp[x]++;
}
// mp存储了 p_i^k_i 格式的素数， (p_i,k_i)
```

## monotonic queue

一个长为n的数组，对于其中每一个长度为m的子区间求最大值:

```c++
deque<pair<int,int>>dq;// val id
for(int i = 0;i<n;i++){
    while(dq.front().second<i-m)dq.pop_front();//下标小于，剔除
    dp[i] = max(dq.front().first,a[i]);
    while(dq.size()&&dq.back().first<=a[i])dq.pop_back();//剔除小于的，val小于，下标也小于,row
    dq.push_back({a[i],i});
}
```

