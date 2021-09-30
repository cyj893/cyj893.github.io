---
title: "알고리즘 정리"
date: 2021-09-05
categories:
  - Algorithm
tags:
  - Algorithm
---

<br></br>
알고리즘들을 정리하고 복습하자
<br></br>

## 알고리즘

### Binary Search
#### Binary Search
```cpp
// now finding 'num'
    int l = 0, h = n;
    int ans = 0;
    while( l <= h ){
        int mid = (l+h) / 2;

        // do something

        if( /*some*/ ){
            l = mid + 1;
            update(ans);
        }
        else h = mid - 1;
    }
    ans = l;
```
종료 조건이 다양하니까 그 때 그 때 다르게
<br></br>

#### stl
```cpp
// now finding 'num'
    auto it = lower_bound(arr.begin(), arr.end(), num);

    if( binary_search(arr.begin(), arr.end(), num) ) 찾음
    else 못 찾음
```
그냥 찾기만 하면 된다면, stl로 O(logN) 사용
set, map에서 `find()`는 O(N)임
<br></br>

### Two Pointer
```cpp
    int l = 0, r = v.size()-1;
    while( l < r ){
        if( 조건 ){
            break;
        }
        else if( 조건 < w ) l++;
        else r--;
    }
```
정렬된 상태에서 양쪽의 값을 더해서 작으면 왼쪽 포인터를 오른쪽으로, 크면 오른쪽 포인터를 왼쪽으로  
<br></br>

### Floyd Warshall
```cpp
    for(int k = 1; k <= n; k++)
        for(int i = 1; i <= n; i++)
            for(int j = 1; j <= n; j++)
                graph[i][j] = min(graph[i][j], graph[i][k] + graph[k][j]);
```

<br></br>

### Dijkstra
```cpp
    for(int i = 1; i <= n; i++){
        dist[i] = INT_MAX;
    }
    priority_queue<P, vector<P>, greater<P>> pq;
    dist[start] = 0;
    pq.push(make_pair(0, start));
    while( pq.size() ){
        int v = pq.top().second;
        int w = pq.top().first;
        pq.pop();

        if( dist[v] < w ) continue;

        for(int i = 0; i < graph[v].size(); i++){
            int u = graph[v][i].second;
            if( dist[u] > w + graph[v][i].first ){
                dist[u] = w + graph[v][i].first;
                pq.push(make_pair(dist[u], u));
            }
        }
    }
```

<br></br>

### Bellman Ford
```cpp
    for(int i = 1; i <= n; i++){
        dist[i] = MAX;
    }
    dist[1] = 0;
    for(int i = 1; i <= n; i++){
        for(int j = 0; j < edges.size(); j++){
            int s = get<0>(edges[j]);
            int e = get<1>(edges[j]);
            int t = get<2>(edges[j]);
            if( dist[s] != MAX && dist[e] > dist[s] + t ){
                dist[e] = dist[s] + t;
                if( i == n ) return true;
            }
        }
    }
    return false;
```
<br></br>

### Union Find
```cpp
struct UnionFind{
	vector<int> parent, ran;
    UnionFind(int n) : parent(n+1), ran(n+1, 1){
    	for(int i = 1; i <= n; i++){
            parent[i] = i;
    	}
    }
    int f(int u){
    	if( u == parent[u] ) return u;
    	parent[u] = f(parent[u]);
        return parent[u];
    }
    bool merg(int u, int v){
    	u = f(u); v = f(v);
        if( u == v ) return false;
        if( ran[u] > ran[v] ) swap(u, v);
        parent[u] = v;
        ran[v] += ran[u];
        ran[u] = 0;
        return true;
    }
};
```
<br></br>

### 최소 신장 트리(MST)
#### Kruskal(Union Find)
```cpp
    UnionFind uf = UnionFind(n);
    int ans = 0;
    while( pq.size() ){
        int c = get<0>(pq.top());
        int a = get<1>(pq.top());
        int b = get<2>(pq.top());
        pq.pop();
        if( uf.merg(a, b) ) ans += c;
    }
    cout << ans << endl;
```
<br></br>

### Manachers Algorithm
```cpp
    string s = "#";
    for(int i = 0; i < t.size(); i++){
        s.append(1, t[i]);
        s.append(1, '#');
    }
    for(int i = 0; i < s.size(); i++){
        int cnt = 0;
        for(int j = 1; j < s.size(); j++){
            if( i-j < 0 || s.size() <= i+j ) break;
            if( s[i-j] != s[i+j] ) break;
            cnt++;
        }
        manachers[i] = cnt;
    }
```
<br></br>

### Topological Sort
```cpp
    for(int i = 0; i < m; i++){
        int a, b;
        cin >> a >> b;
        v[a].push_back(b);
        ind[b]++;
    }
    queue<int> q;
    for(int i = 1; i <= n; i++){
        if( ind[i] == 0 ) q.push(i);
    }
    while( q.size() ){
        int now = q.front();
        q.pop();
        for(int i = 0; i < v[now].size(); i++){
            int nx = v[now][i];
            ind[nx]--;
            if( ind[nx] == 0 ) q.push(nx);
        }
    }
```
만약 인덱스가 빠른 순이면 priority queue 사용
<br></br>

### Trie
```cpp
vector<string> v;

struct Trie{
    bool fin;
    string val;
    map<string, Trie*> nodes;
    vector< pair<string, Trie*> > sortedNodes;
    Trie(string s){
        fin = false;
        val = s;
    }
    void Tinsert(int i){
        if( nodes.count(v[i]) == 0 ) nodes[v[i]] = new Trie(v[i]);
        if( v.size()-1 == i ) fin = true;
        else nodes[v[i]]->Tinsert(i+1);
    }
    void Tsort(){
        sortedNodes = vector< pair<string, Trie*> >(nodes.begin(), nodes.end());
        sort(sortedNodes.begin(), sortedNodes.end());
    }
    void Tprint(int d){
        Tsort();
        for(int i = 0; i < sortedNodes.size(); i++){
            for(int j = 0; j < d; j++){
                cout << "--";
            }
            cout << sortedNodes[i].first << endl;
            sortedNodes[i].second->Tprint(d+1);
        }
    }
};

// in main()
    Trie trie = Trie("");

    // push to vector
    trie.Tinsert(0);
    // clear vector

    trie.Tprint(0);
```
<br></br>

### Segment Tree

#### 구간 합
```cpp
long long makeSeg(int a, int b, int now){
    if( a == b ) return segtree[now] = nums[a];
    int mid = (a+b) / 2;
    segtree[now] = makeSeg(a, mid, now*2) + makeSeg(mid+1, b, now*2+1);
    return segtree[now];
}

long long sumSeg(int a, int b, int now, int l, int r){
    if( r < a || b < l ) return 0;
    if( l <= a && b <= r ) return segtree[now];
    int mid = (a+b) / 2;
    return sumSeg(a, mid, now*2, l, r) + sumSeg(mid+1, b, now*2+1, l, r);
}

void updateSeg(int a, int b, int now, int ind, long long change){
    if( ind < a || b < ind ) return;
    segtree[now] += change;
    if( a == b ) return;
    int mid = (a+b) / 2;
    updateSeg(a, mid, now*2, ind, change);
    updateSeg(mid+1, b, now*2+1, ind, change);
}

// in main()
    int h = ceil(log2(n));
    segtree.assign(1<<(h+1), 0);
    makeSeg(0, n-1, 1);

    long long change = val - nums[b];
    updateSeg(0, n-1, 1, b, change);
    nums[b] = c;

    cout << sumSeg(0, n-1, 1, b, c) << '\n';
```

#### 구간 곱
```cpp
long long makeSeg(int a, int b, int now){
    if( a == b ) return segtree[now] = nums[a];
    int mid = (a+b) / 2;
    segtree[now] = makeSeg(a, mid, now*2) * makeSeg(mid+1, b, now*2+1);
    segtree[now] %= MAX;
    return segtree[now];
}

long long sumSeg(int a, int b, int now, int l, int r){
    if( r < a || b < l ) return 1;
    if( l <= a && b <= r ) return segtree[now];
    int mid = (a+b) / 2;
    long long s1 = sumSeg(a, mid, now*2, l, r) % MAX;
    long long s2 = sumSeg(mid+1, b, now*2+1, l, r) % MAX;
    return (s1 * s2) % MAX;
}

long long updateSeg(int a, int b, int now, int ind, long long val){
    if( ind < a || b < ind ) return segtree[now];
    if( a == b ) return segtree[now] = val;
    int mid = (a+b) / 2;
    long long u1 = updateSeg(a, mid, now*2, ind, val);
    long long u2 = updateSeg(mid+1, b, now*2+1, ind, val);
    return segtree[now] = (u1*u2) % MAX;
}
```

<br></br>

### 좌표 압축
```cpp
    for(int i = 1; i <= n; i++){
        cin >> v[i];
        comp.push_back(v[i]);
    }
    sort(comp.begin(), comp.end());
    comp.erase(unique(comp.begin(), comp.end()), comp.end());
    for(int i = 1; i <= n; i++){
        v[i] = lower_bound(comp.begin(), comp.end(), v[i]) - comp.begin() + 1;
    }
```

<br></br>

### LCA, 두 정점 사이 거리
```cpp
vector<P> tree[40001];
int depth[40001];
int ac[40001][16];
int dist[40001];
int max_level;

void getTree(int now, int pre, int d){
    depth[now] = depth[pre] + 1;
    ac[now][0] = pre;
    dist[now] = d;

    for(int i = 1; i <= max_level; i++){
        int tmp = ac[now][i - 1];
        ac[now][i] = ac[tmp][i - 1];
    }

    for(int i = 0; i < tree[now].size(); i++){
        int nd = tree[now][i].first;
        int nx = tree[now][i].second;
        if( nx != pre ) getTree(nx, now, d + nd);
    }
}

int getlca(int a, int b){
    if( depth[a] != depth[b] ){
        if( depth[a] > depth[b] ) swap(a, b);
        for(int i = max_level; i >= 0; i--){
            if( depth[a] <= depth[ac[b][i]] ){
                b = ac[b][i];
            }
        }
    }
    int ret = a;
    if( a != b ){
        for(int i = max_level; i >= 0; i--){
            if( ac[a][i] != ac[b][i] ){
                a = ac[a][i];
                b = ac[b][i];
            }
            ret = ac[a][i];
        }
    }
    return ret;
}

// in main()

    max_level = (int)floor(log2(n));

    for(int i = 0; i < n-1; i++){
        int a, b, c;
        cin >> a >> b >> c;
        tree[a].push_back(make_pair(c, b));
        tree[b].push_back(make_pair(c, a));
    }

    depth[0] = -1;
    getTree(1, 0, 0);


    // 두 정점 사이 거리
        int lca = getlca(a, b);
        dist[a] + dist[b] - 2*dist[lca]
```

<br></br>

### KMP
```cpp
    int psz = p.size();
    vector<int> pi(psz, 0);
    int start = 1, matched = 0;
    while( start + matched < psz ){
        if( p[start + matched] == p[matched] ){
            matched++;
            pi[start + matched - 1] = matched;
        }
        else{
            if( matched == 0 ) start++;
            else{
                start += matched - pi[matched - 1];
                matched = pi[matched - 1];
            }
        }
    }

    vector<int> ans;
    matched = 0;
    for(int i = 0; i < t.size(); i++){
        while( matched > 0 && t[i] != p[matched] ){
            matched = pi[matched - 1];
        }
        if( t[i] == p[matched] ){
            matched++;
            if( matched == psz ){
                ans.push_back(i - psz + 2);
                matched = pi[matched - 1];
            }
        }
    }
```

<br></br>

### LIS
#### 이분탐색
```cpp
pair<int, int> line[100001];
pair<int, int> tracking[100001];
vector<int> ind;

int main(){
    ios::sync_with_stdio(false);
    cin.tie(NULL);

    int n;
    cin >> n;
    for(int i = 0; i < n; i++){
        int a, b;
        cin >> a >> b;
        line[i] = make_pair(a, b);
    }

    sort(line, line + n);

    ind.push_back(line[0].second);
    tracking[0] = make_pair(1, line[0].first);
    for(int i = 1; i < n; i++){
        int a = line[i].first;
        int b = line[i].second;
        if( b > ind.back() ){
            ind.push_back(b);
            tracking[i] = make_pair(ind.size(), a);
        }
        else{
            auto it = lower_bound(ind.begin(), ind.end(), b);
            *it = min(*it, b);
            tracking[i] = make_pair(it-ind.begin()+1, a);
        }
    }

    cout << ind.size() << '\n';

    stack<int> st;
    int t = ind.size();
    for(int i = n-1; i >= 0; i--){
        if( tracking[i].first == t ){
            st.push(tracking[i].second);
            t--;
        }
    }
    while( st.size() ){
        cout << st.top() << '\n';
        st.pop();
    }
}
```

<br></br>

#### 세그먼트 트리
```cpp
int n;
pair<int, int> v[1000001];
vector<int> segtree;

int maxSeg(int a, int b, int now, int l, int r){
    if( r < a || b < l ) return 0;
    if( l <= a && b <= r ) return segtree[now];
    int mid = (a+b) / 2;
    return max(maxSeg(a, mid, now*2, l, r), maxSeg(mid+1, b, now*2+1, l, r));
}

int updateSeg(int a, int b, int now, int ind, int change){
    if( ind < a || b < ind ) return 0;
    if( a == b ) return segtree[now] = change;
    int mid = (a+b) / 2;
    return segtree[now] = max({segtree[now], updateSeg(a, mid, now*2, ind, change), updateSeg(mid+1, b, now*2+1, ind, change)});
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(NULL);

    cin >> n;
    for(int i = 0; i < n; i++){
        int a;
        cin >> a;
        v[i] = make_pair(a, i);
    }
    sort(v, v + n, [](pair<int, int> p, pair<int, int> q) {
		if (p.first != q.first) return p.first < q.first;
		return p.second > q.second;
	});

    int h = ceil(log2(n));
    segtree.assign(1<<(h+1), 0);

    stack< pair<int, int> > st;
    for(int i = 0; i < n; i++){
        int a = maxSeg(0, n-1, 1, 0, v[i].second) + 1;
        updateSeg(0, n-1, 1, v[i].second, a);
        st.push(make_pair(a, v[i].first));
    }
    int m = maxSeg(0, n-1, 1, 0, n-1);
    cout << m << endl;

    stack<int> st2;
    m++;
    while( st.size() ){
        while( st.size() && st.top().first != m-1 ){
            st.pop();
        }
        st2.push(st.top().second);
        m = st.top().first;
        if( m == 1 ) break;
    }

    while( st2.size() ){
        cout << st2.top() << ' ';
        st2.pop();
    }
}
```

<br></br>

### DAG에서 최장 경로
```cpp
vector<P> graph[10001];
vector<P> revgraph[10001];
int ind[10001];
int times[10001];

int main(){
    ios::sync_with_stdio(false);
    cin.tie(NULL);

    int n, m;
    cin >> n >> m;
    for(int i = 0; i < m; i++){
        int a, b, c;
        cin >> a >> b >> c;
        graph[a].push_back(make_pair(c, b));
        revgraph[b].push_back(make_pair(c, a));
        ind[b]++;
    }
    int start, arrival;
    cin >> start >> arrival;

    queue<int> q;
    q.push(start);
    while( q.size() ){
        int now = q.front();
        q.pop();
        for(int j = 0; j < graph[now].size() ; j++){
            int c = graph[now][j].first;
            int nx = graph[now][j].second;
            ind[nx]--;
            times[nx] = max(times[nx], times[now] + c);
            if( ind[nx] == 0 ) q.push(nx);
        }
    }
    cout << times[arrival] << '\n';

    q.push(arrival);
    ind[arrival] = 1;
    int cnt = 0;
    while( q.size() ){
        int now = q.front();
        q.pop();

        if( now == start ) break;

        for(int j = 0; j < revgraph[now].size() ; j++){
            int c = revgraph[now][j].first;
            int nx = revgraph[now][j].second;
            if( times[now] - c == times[nx] ){
                cnt++;
                if( ind[nx] == 0 ){
                    ind[nx] = 1;
                    q.push(nx);
                }
            }
        }
    }
    cout << cnt << '\n';
}
```

<br></br>

### SCC
```cpp
vector<int> graph[10001];
vector<int> revgraph[10001];
int visited[10001];
priority_queue<P, vector<P>, less<>> pq;
vector< vector<int> > ans;
int d;

void func(int now){
    for(int i = 0; i < graph[now].size(); i++){
        int nx = graph[now][i];
        if( visited[nx] == 0 ){
            visited[nx] = 1;
            d++;
            func(nx);
        }
    }
    d++;
    pq.push(make_pair(d, now));
}

void func2(int now, int ind){
    for(int i = 0; i < revgraph[now].size(); i++){
        int nx = revgraph[now][i];
        if( visited[nx] == 0 ){
            visited[nx] = 1;
            ans[ind].push_back(nx);
            func2(nx, ind);
        }
    }
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(NULL);

    int v, e;
    cin >> v >> e;
    for(int i = 0; i < e; i++){
        int a, b;
        cin >> a >> b;
        graph[a].push_back(b);
        revgraph[b].push_back(a);
    }

    d = 1;
    for(int i = 1; i <= v; i++){
        if( visited[i] ) continue;
        visited[i] = 1;
        func(i);
    }

    for(int i = 1; i <= v; i++){
        visited[i] = 0;
    }
    int ind = 0;
    while( pq.size() ){
        int now = pq.top().second;
        pq.pop();

        if( visited[now] ) continue;

        vector<int> t = {now};
        ans.push_back(t);
        visited[now] = 1;
        func2(now, ind);
        ind++;
    }
}
```

<br></br>


## 수식
### 경우의 수
#### 조합
```cpp
nCr[i][j] = n! / (r! * (n-r)!);

nCr[i][j] = nCr[i-1][j-1] + nCr[i-1][j];

(a+b)^n = nCr[n][0]*a^0*b^n + nCr[n][1]*a^1*b^n-1 + ... + nCr[n][n]*a^n*b^0

nCr[n][0] + nCr[n][1] + ... + nCr[n][n] = 2^n
nCr[n][0] + nCr[n][2] + ... = 2^(n-1)
```

#### 같은 것이 있는 순열
```cpp
ex) aaabb => 5! / (3! * 2!), 최단거리 길찾기
n! / (an! * bn! * ... * zn!)
```

#### 중복 조합
```cpp
nHr = (n+r-1)Cr
```
<br></br>

### GCD
```cpp
int gcd(int a, int b){
    if( b > a ) return gcd(b, a);
    if( a%b == 0 ) return b;
    return gcd(b, a%b);
}
```
<br></br>

### 피보나치
```cpp
| Fn+1 Fn   |       | 1 1 |
| Fn   Fn-1 |       | 1 0 |
```
행렬 곱으로 나타내기
#### O(logN)
```cpp
struct F{
    long long a, b, c, d;
};
F one = F{1, 1, 1, 0};

F func(F f1, F f2){
    F t;
    t.a = f1.a * f2.a + f1.b * f2.c;
    t.b = f1.a * f2.b + f1.b * f2.d;
    t.c = f1.c * f2.a + f1.d * f2.c;
    t.d = f1.c * f2.b + f1.d * f2.d;
    t.a %= 1000000007;
    t.b %= 1000000007;
    t.c %= 1000000007;
    t.d %= 1000000007;
    return t;
}

F fibo(F f, long long d){
    if( d == 1 ) return one;
    if( d % 2 ) return func(fibo(f, d-1), one);
    F t = fibo(f, d/2);
    return func(t, t);
}

// in main()
    if( n == 0 ) cout << 0 << '\n';
    else if( n == 1 ) cout << 1 << '\n';
    else{
        F f = F{1, 1, 1, 0};
        f = fibo(f, n);
        cout << f.b % 1000000007 << endl;
    }
```
<br></br>

### 빠른 pow
```cpp
long long mypow(long long a, int b){
    if( b == 0 ) return 1;
    if( b == 1 ) return a;
    if( b % 2 ) return a*mypow(a, b-1) % MOD;
    long long aa = mypow(a, b/2) % MOD;
    return aa*aa % MOD;
}
```
<br></br>

### 모듈러 곱셈의 역원
```cpp
a^(MOD-2) = a의 곱셈의 역원
```

<br></br>

### 소인수분해
```cpp
    for(long long i = 2; i*i <= n; i++){
        long long cnt = 0;
        while( n % i == 0 ){
            n /= i;
            cnt++;
        }
        primes[i] = cnt;
    }
    if( n - 1 ) primes[n-1] = 1;
```

<br></br>

### 오일러 파이(서로소의 개수)
```cpp
phi(n) = (A^a - A^(a-1)) * (B^b - B^(b-1)) * ... * (Z^z - Z^(z-1))
       = A^(a-1)*(A-1) * B^(b-1)*(B-1) * ... * Z^(z-1)*(Z-1)

    for(long long i = 2; i*i <= n; i++){
        long long cnt = 0;
        while( n % i == 0 ){
            n /= i;
            cnt++;
        }
        if( cnt ){
            ans *= pow(i, cnt-1);
            ans *= i - 1;
        }
    }
    if( n - 1 ) ans *= n - 1;
```
<br></br>

### Signed Area, CCW
#### 다각형 넓이 구하기
```cpp
(다각형을 이루는 순서대로일 때)
| x1 x2 ... xn | > + x1*yn + x2*y1 + ... + xn*yn-1
| y1 y2 ... yn | > - x1*y2 - x2*y3 - ... - xn*y1
ans = sarea / 2;
```

#### 교차 판단
```cpp
int ccw(pair<long long, long long> &a,
          pair<long long, long long> &b,
          pair<long long, long long> &c){
    long long x1 = a.first;
    long long y1 = a.second;
    long long x2 = b.first;
    long long y2 = b.second;
    long long x3 = c.first;
    long long y3 = c.second;
    long long ret = x2*y1 + x3*y2 + x1*y3 - (x1*y2 + x2*y3 + x3*y1);
    if( ret < 0 ) return 1;
    if( ret == 0 ) return 0;
    return -1;
}
```
<br></br>

### 벌집 인덱스
![1](/img/Algorithm/14/1.png)  
<br></br>

---

<br></br>
