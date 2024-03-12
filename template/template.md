# dp

构造。

dp的边界条件可以不在循环的头中显性表示，而是通过**初始化**时的 INF -INF 加上 if continue语句简化处理。

dp的寻找构造一般不包含无穷的情况（概率dp）

# binary search



寻找边界任务：对于左边 < , 右边 > :

```c++
int l = 1,r = n;
while(l<r-1){ //条件如下，否则进入死循环
	int mid = (l+r)>>1;//尽量用这个>>而不是除
    if(check(mid)==left)l = mid
    else r = mid
}
```

# graph

## shortest path

### dijkstra

```c++
void dijkstra(vector<vector<pair<int, int>>>& graph, int start, vector<int>& distance) {
    int numNodes = graph.size();
    distance.assign(numNodes, INF);


    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;

    distance[start] = 0;
    pq.push({0, start});

    while (!pq.empty()) {
        int u = pq.top().second;
        pq.pop();

        for (const auto& edge : graph[u]) {
            int v = edge.first;
            int weight = edge.second;
            // Relaxation step
            if (distance[u] + weight < distance[v]) {
                distance[v] = distance[u] + weight;
                pq.push({distance[v], v});
            }
        }
    }
}
```

variation:

dijkstra的思想可以用在许多不同格式的最小值上，对于一个结构体，以pair为例，第一个值为当前的cost，第二个为节点的id: u，那么可以有如下的变种：

```c++
    priority_queue<pair<int,int>,vector<pair<int,int>>,greater<pair<int,int>>>pq;
    //初始化 pq.push xxxx
    while(!pq.empty()){
        auto [cost,u] = pq.top();
        pq.pop();
        
        if(dist[u]!=cost)continue;//这一句说明当前的pair node 是冗余的，因为新cost必定比dist大
        
        if(is_target){//若是一个目标直接输出即可，如果为1个点到所有边的话最后输出也可。
            cout<<cost<<"\n";return;
        }
        for(auto [v,val]:edges[u]){//遍历边 这里边的格式为 {u,val}分别代表邻接节点以及特征值
            //对于 u 到 v 的一条边，或者说从开始到v的new cost ，可以通过val以及对应的预处理获得
            ncost = handle(val,cost)//可能与当前的cost等其他变量也有关。
            
          	//这里同样根据新cost跟新dist
            if(ncost < dist[v]){
                dist[v] = ncost +1;
                pq.push({dist[v],v});
            }
        }
    }
```

时间复杂度为
$$
O((V+E)(logV+handle的时间)
$$



上述还有一种化简的方式，利用pq的特性，最小堆：

```c++
if(dist[u]!=-1)continue;
else dist[u] = cost;
```

但是我仍未完全明白。

### bipartie graph traversal

如果两者之间的边关系若直接建立较多，例如 ai,aj之间有边iff gcd(ai,aj)>1，这时直接建立n^2会暴，不妨建立二部图，从ai 到其素因数。

```c++
for(auto c: b[a[i]]){
    g[i].push_back(n+c);
    g[n+c].push_back(i);
}
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



# game theory



## sg fucntion

[博弈论 | 详解搞定组合博弈问题的SG函数 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/157731188)
$$
\text{SG}(S) = \text{mex}\left(\{\text{SG}(T) \mid T \text{ is a valid next position from } S\}\right)
$$
通常通过子问题的异或得到最终的结果。
$$
\text{SG}(S_1 + S_2) = \text{SG}(S_1) \oplus \text{SG}(S_2)
$$


# number theory

## prime number 

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
        // i % pri[j] == 0
        // 换言之，i 之前被 pri[j] 筛过了
        // 由于 pri 里面质数是从小到大的，所以 i乘上其他的质数的结果一定会被
        // pri[j]的倍数筛掉，就不需要在这里先筛一次，所以这里直接 break
        // 掉就好了
        break;
      }
    }
  }
}
```



### prime detection

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



# constructive 

think with balls , use counter example or proofs to testify.

# tools

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

### 封装版本

```c++
using ll = long long;
const ll mod = 998244353;
template<class Info,
    class Merge = plus<Info>>
struct SegmentTree {
    const int n;
    const Merge merge;
    vector<Info> info;
    SegmentTree(int n) : n(n), merge(Merge()), info(4 << int(log2(n))) {}
    SegmentTree(vector<Info> init) : SegmentTree(init.size()) {
        function<void(int, int, int)> build = [&](int p, int l, int r) {
            if (r - l == 1) {
                info[p] = init[l];
                return;
            }
            int m = (l + r) / 2;
            build(2 * p, l, m);
            build(2 * p + 1, m, r);
            pull(p);
        };
        build(1, 0, n);
    }
    void pull(int p) {
        info[p] = merge(info[2 * p], info[2 * p + 1]);
    }
    void modify(int p, int l, int r, int x, const Info &v) {
        if (r - l == 1) {
            info[p] = v;
            return;
        }
        int m = (l + r) / 2;
        if (x < m) {
            modify(2 * p, l, m, x, v);
        } else {
            modify(2 * p + 1, m, r, x, v);
        }
        pull(p);
    }
    void modify(int p, const Info &v) {
        modify(1, 0, n, p, v);
    }
    Info rangeQuery(int p, int l, int r, int x, int y) {
        if (l >= y || r <= x) {
            return Info();
        }
        if (l >= x && r <= y) {
            return info[p];
        }
        int m = (l + r) / 2;
        return merge(rangeQuery(2 * p, l, m, x, y), rangeQuery(2 * p + 1, m, r, x, y));
    }
    Info rangeQuery(int l, int r) {
        return rangeQuery(1, 0, n, l, r);
    }
};
struct mat {
    ll a[2][2]={0};
};
mat operator+(const mat&a,const mat&b)
{
    mat ret;
    for(int i = 0;i<2;i++){
        for(int j = 0;j<2;j++){
            for(int k = 0;k<2;k++){
                ret.a[i][j] = (ret.a[i][j]+a.a[i][k]*b.a[k][j]%mod)%mod;
            }
        }
    }
    return ret;
}
```



### 主席树：

支持不同根的更新：

```cpp
const int N = 1e5 + 5;
struct info{
    int sum = 0;
};

info operator+(const info &a, const info &b){
	return {a.sum + b.sum};
}

struct Node{
    info info;
    int l, r;
}tr[N * 90];
int root[N];
int idx;

void build(int &u, int l, int r, const vector<info> &init){
    if (!u) u = ++idx;
    if (l == r){
        tr[u].info = init[r - 1];
        return;
    }
    int mid = (l + r) / 2;
    build(tr[u].l, l, mid, init);
    build(tr[u].r, mid + 1, r, init);
    tr[u].info = tr[tr[u].l].info + tr[tr[u].r].info;
}

void update(int &now, int pre, int l, int r, int x, int v){
    tr[now = ++idx] = tr[pre];
    if (l == r){
        tr[now].info.sum += v;
        return;
    }
    int mid = (l + r) / 2;
    if (x <= mid) update(tr[now].l, tr[pre].l, l, mid, x, v);
    else update(tr[now].r, tr[pre].r, mid + 1, r, x, v);
    tr[now].info = tr[tr[now].l].info + tr[tr[now].r].info;
}

info query(int u, int l, int r, int L, int R){
    if (r < L || l > R){
        return info();
    }
	if (l >= L && r <= R){
	 	return tr[u].info;
	}
	int mid = (l + r) / 2;
	return query(tr[u].l, l, mid, L, R) + query(tr[u].r, mid + 1, r, L, R);
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

带权并查集:



## LCA

找树上两点路径上边权重最小值为例

```c++
int d[N],f[N][20],c[N][20];//20 adapt to lg2(N)
void dfs(int u, int fa) {
    d[u] = d[fa] + 1;
    for (auto x : g[u]) {
        int v = x.first;

        if (v == fa)
            continue;
        dfs(v, u);
        f[v][0] = u;
        c[v][0] = x.second;
    }
}
int lca(int x,int y)
{
	if(d[x]>d[y])swap(x,y);
	int t = d[y] - d[x];
	int ans = inf;
	for(int j=0;t;t>>=1,j++){
		if(t&1){
			ans = min(ans,c[y][j]);
			y = f[y][j];
		}
	}
	if(x==y)return ans;
	
	for(int j = K-1;j>=0;j--){
		if(f[x][j]!=f[y][j]){
			ans = min(ans,min(c[x][j],c[y][j]));
			x = f[x][j];
			y = f[y][j];
		}
	}
	ans = min(ans,min(c[x][0],c[y][0]));
	return ans;
}
// main当中需要进行一下初始化:
int main
{
    
    for (int i = 1; (1 << i) <= n; i++) {
        for (int j = 1; j <= n; j++) {
            f[j][i] = f[f[j][i - 1]][i - 1];
            c[j][i] = min(c[j][i - 1], c[fl[j][i - 1]][i - 1]);
        }
    }
}
```



## MODINT

template given:

```c++
template<int MOD>
struct ModInt{
    unsigned x;
    ModInt(){x=0;}
    ModInt(signed sig):x(sig){}
    ModInt(signed long long sig): x(sig%MOD) {}
    int get() const {return (int)x;}
    ModInt pow(long long p){ModInt res = 1,a = *this;while(p){if(p&1)res*=a;a*=a;p>>=1;}return res;}

    ModInt operator+=(ModInt that){if((x+=that.x)>=MOD)x-=MOD;return *this;}
    ModInt operator-=(ModInt that){if((x+=MOD-that.x)>=MOD)x-=MOD;return *this;}
    ModInt operator*=(ModInt that){x=(unsigned long long)x*that.x%MOD;return *this;}
    ModInt operator/=(ModInt that){return (*this)*=that.pow(MOD-2);}

    ModInt operator+(ModInt that)const {return ModInt(*this)+=that;}
    ModInt operator-(ModInt that)const {return ModInt(*this)-=that;}
    ModInt operator*(ModInt that)const {return ModInt(*this)*=that;}
    ModInt operator/(ModInt that)const {return ModInt(*this)/=that;}
    bool operator<(ModInt that) const {return x<that.x;}
    friend ostream&operator<<(ostream&os,ModInt a){os<<a.x;return os;}
};
```

## STRING SUFFIX STRUCTURE

### AC automata

```c++
int sz,ch[M][26],f[M],cnt[M];
void init()
{
	sz = 1;
	memset(ch[0],0,sizeof(ch[0]));
	cnt[0] = 0;
}
void insert(const string&s)
{
	int len = s.length();
	int u = 0;
	for(int i = 0;i<len;i++){
		int c = s[i]-'a';
		if(!ch[u][c]){
			memset(ch[sz],0,sizeof(ch[sz]));
			cnt[sz] = 0;
			ch[u][c] = sz++;
		}
		u = ch[u][c];
	}
	++cnt[u];
}
void getfail()
{
	queue<int>q;
	f[0] = 0;
	for(int c = 0;c<26;c++){
		int u = ch[0][c];
		if(u){
			f[u] = 0;
			q.push(u);
		}
	}
	while(q.size()){
		int r = q.front();q.pop();
		for(int c = 0;c<26;c++){
			int u = ch[r][c];
			if(!u){
				ch[r][c] = ch[f[r]][c];continue;
			} 
			q.push(u);
			int v = f[r];
			while(v && !ch[v][c])v = f[v];
			f[u] = ch[v][c];
			cnt[u]+=cnt[f[u]];
		}
	}
}
```



### SA

有超时风险（1e6 700ms）

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



## NTT (POLY)

**这个板子耗时较长** 

```c++
const int N = 2E5+5;
const ll mod = 998244353;
ll qpow(ll a,ll b)
{
    ll ret = 1;
    while(b){
        if(b&1) ret = ret*a%mod;
        b>>=1,a=a*a%mod;
    }
    return ret;
}
ll inv(ll x){
    return qpow(x,mod-2);
}
const ll G = 3,GI = inv(3);

void NTT(int len, ll * a, int coe)
{
    for (int mid = 1; mid < len; mid <<= 1)
    {
        ll Wn = qpow(((coe == 1) ? G : GI), (mod - 1) / (mid << 1));
        for (int i = 0; i < len; i += mid << 1)
        {
            ll W = 1;
            for (int j = 0; j < mid; j++, W = (W * Wn) % mod)
            {
                ll x = a[i + j], y = (a[i + j + mid] * W) % mod;
                a[i + j] = (x + y) % mod;
                a[i + j + mid] = (x - y + mod) % mod;
            }
        }
    }
}
ll rev[N], c[N];
void mul(int len, ll * a, ll * b)
{
    for (int i = 1; i < len; i++) rev[i] = (rev[i >> 1] >> 1) + ((i & 1) ? (len >> 1) : 0);
    for (int i = 0; i < len; i++) if (i < rev[i]) swap(a[i], a[rev[i]]); NTT(len, a, 1);
    for (int i = 0; i < len; i++) if (i < rev[i]) swap(b[i], b[rev[i]]); NTT(len, b, 1);
    for (int i = 0; i <= len; i++) a[i] = (a[i] * b[i]) % mod;
    for (int i = 0; i < len; i++) if (i < rev[i]) swap(a[i], a[rev[i]]); NTT(len, a, -1);
    ll invl = inv(len);
    for (int i = 0; i <= len; i++) c[i] = (a[i] * invl) % mod;
}
```

调用时：

```c++
while(len<=length*2)len<<=1;//保证计算时不溢出
mul(len,a,b);
```



**下面的板子快一些：**

```c++
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
constexpr ll g=3;
ll maxn,rev[N],w[N];
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
		if(i==2048) for(int j=0;j<maxn;j++) a[j]%=mod;//mod次数优化
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

调用时

```c++
Poly a(n),b(n),c(n);
c = mul(a,b);
```



### 分治NTT

[码题集OJ-课件设计 (matiji.net)](https://www.matiji.net/exam/brushquestion/16/4347/179CE77A7B772D15A8C00DD8198AAC74)

[码题集OJ-随机序列逆序数 (matiji.net)](https://www.matiji.net/exam/brushquestion/6/4347/179CE77A7B772D15A8C00DD8198AAC74)

分治ntt:

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

给一版直接用的:

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

T can be defined so that operator + can be changed into swap and ...

this is an example:

```c++
struct Max {
    int v;
    Max(int x = -1E9) : v{x} {}
    
    Max &operator+=(Max a) {
        v = std::max(v, a.v);
        return *this;
    }
};
```

so when we want to store the current max from prefix of x :

we can define this:

```c++
Fenwick<Max>fen(n)//or n+1 depends
```



## monotonic stack

保证元素始终单调递增或者单调递减。

用处：

例如，对于一个数组 a,遍历里面的元素，对于当前的元素a[i]，每次需要从i开始遍历到0找到第一个满足a[j]<a[i]的下表j：

那么对于这个问题，我们可以使用单调递增栈来维护。

单调栈可以保证最靠近性满足条件，可以用于dp当中的减少时间复杂度。

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



## Euler Phi

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



## trie

字典树求异或最大值

```c++
const int N = 2E6;
//trie tree
int cnt,trie[N][2];
const int LG = 22;
void insert(int x)
{
    for(int i = LG,p=0;i>=0;i--){
        int u = (x>>i)&1;
        if(!trie[p][u])trie[p][u] = ++cnt;
        p = trie[p][u];
    }
}
int query(int x)
{
    for(int i = LG,p=0;i>=0;i--){
        int u = (x>>i)&1;
        if(trie[p][u^1])u^=1;
        x ^= u<<i;
        p = trie[p][u];
    }
    return x;
}
```

## 树哈希求同构

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



# idle

顺序vector求区间内的值，如[l,r] (前提是vector需要排好序)

```c++
upper_bound(v.begin(),v.end(),r)-lower_bound(v.begin(),v.end(),l);
```





quick ways to get the intverval of same value of the function 
$$
f(x) = [\frac{n}{x}] ~~~~~~~~~x = {1,2,...,n}
$$

```c++
vector<pair<int,int>>p;
for (int left = 1, right; left < n; left = right + 1){
    int C = (n + left - 1) / left;
    right = (n + C - 1 - 1) / (C - 1) - 1;
    p.push_back({left,right});
}
```



区间遍历可以使用born dead 方法可以实时检测区间的覆盖情况。



##### O(N) 时间排序

前提是数据大小有限，不能是1e9

```c++
int main()
{
    int n = 10;
    int cnt = 10;
    vector<int>len(cnt+1);
    len = {1,1,6,3,2,4,5,6,7,8,9};
    vector<int>buc(n),id(cnt);
    for(int i = 1;i<=cnt;i++)buc[len[i]]++;
    for(int i = 1;i<=n;i++)buc[i]+=buc[i-1];
    for(int i = cnt;i;i--)id[buc[len[i]]--] = i;
    //关键，len(i)<len(j),buc[len[i]]<buc[len[j]]
    for(int i = cnt;i;i--){
        cout<<len[id[i]]<<" ";
    }
}
```



pair从小到大排序后，存后缀的最大值的下标。O(n)

```c++
for(int i = n-1 ;i>=0;i--){
    p[i] = i;
    if(i+1<n&&val[p[i]]<val[p[i+1]]){
        p[i] = p[i+1];
    }
}
```



