# フォーメーション飛行の改善案

## 概要

フォーメーション飛行において観察された3つの問題点と、それらを統合的に解決するための実装計画をまとめます。

## 問題点と改善案

### P1: フォロワー間の交差によるデッドロック

**問題**: 1,2が0の相対距離に対して交差している場合、CBFの制約により、デッドロック（間を通り過ぎれない）が発生する。

**改善案 (S1)**: 1がリーダーの右側、2が左側などの固定ルールをなくし、リーダーに対して最終的にフォーメーションを組めていればどっちのエージェントがどっちにゴールを設定してもOKに変更。

### P2: リーダーとフォロワーの優先順位問題

**問題**: 0と1,2のゴールへの優先順位が同等なため、2にぶつからないように0がゴールから遠ざかる方向に避けると、2の目標地点はさらに遠くなり、永遠にフォーメーションが達成されない。

**改善案 (S2)**: 0が王様で、1,2は王様の進路を邪魔してはならない（つまり、0は障害物に対しては衝突回避をするが、エージェントに対しては衝突回避を行わない）ように優先順位をつける。

### P3: 障害物によるフォーメーション維持不能

**問題**: そもそも障害物の位置関係的にフォーメーションが保てない幅だとデッドロックが発生する。

**改善案 (S3)**: 
- フォロワーの目標地点（リーダーからのオフセット）にも安全性の項やCBFを適用させる
- または、先に通過するリーダーが、通り過ぎた障害物に対して、これはフォロワーが通れないなと検知したらオフセットを動的に変更する

## 実装計画

### 1. 柔軟なフォーメーション割り当て（P1対応）

#### 1.1 概要

現在の実装では、`formation_offsets`の順序が固定されており、エージェント1が最初のオフセット、エージェント2が2番目のオフセットに割り当てられています。これを動的に割り当て可能にします。

#### 1.2 変更箇所

**ファイル**: `gcbfplus/env/double_integrator.py`

##### 1.2.1 PARAMSの拡張

```python
PARAMS = {
    # ... 既存のパラメータ ...
    "formation_mode": False,
    "formation_offsets": None,
    "formation_flexible_assignment": False,  # 新規追加: 柔軟な割り当てを有効化
    "formation_min_distance": 0.1,  # 新規追加: フォーメーション達成とみなす最小距離
}
```

##### 1.2.2 フォーメーション割り当て関数の追加

```python
def assign_formation_offsets(
    self, 
    leader_pos: Array, 
    follower_positions: Array, 
    formation_offsets: Array
) -> Array:
    """
    フォロワーの現在位置に基づいて、最適なオフセット割り当てを決定する。
    
    Args:
        leader_pos: リーダーの位置 [x, y]
        follower_positions: フォロワーの現在位置 [n_followers, 2]
        formation_offsets: 利用可能なオフセット [n_offsets, 2]
    
    Returns:
        各フォロワーに割り当てるオフセットのインデックス [n_followers]
    """
    n_followers = follower_positions.shape[0]
    n_offsets = formation_offsets.shape[0]
    
    # 各フォロワーと各オフセットの組み合わせについて、目標位置を計算
    target_positions = leader_pos[None, :] + formation_offsets[:, None, :]  # [n_offsets, n_followers, 2]
    target_positions = jnp.transpose(target_positions, (1, 0, 2))  # [n_followers, n_offsets, 2]
    
    # 各フォロワーから各目標位置への距離を計算
    distances = jnp.linalg.norm(
        follower_positions[:, None, :] - target_positions, 
        axis=-1
    )  # [n_followers, n_offsets]
    
    # ハンガリアンアルゴリズム（または貪欲法）で最適な割り当てを決定
    # 簡易実装として貪欲法を使用
    assignment = jnp.zeros(n_followers, dtype=jnp.int32)
    used_offsets = jnp.zeros(n_offsets, dtype=jnp.bool_)
    
    def assign_one(carry, follower_idx):
        assignment, used_offsets, distances = carry
        # 未使用のオフセットの中で、このフォロワーに最も近いものを選択
        masked_distances = jnp.where(
            used_offsets[:, None],
            jnp.inf,
            distances[follower_idx, :]
        )
        best_offset_idx = jnp.argmin(masked_distances)
        assignment = assignment.at[follower_idx].set(best_offset_idx)
        used_offsets = used_offsets.at[best_offset_idx].set(True)
        return (assignment, used_offsets, distances), None
    
    (assignment, _, _), _ = jax.lax.scan(
        assign_one,
        (assignment, used_offsets, distances),
        jnp.arange(n_followers)
    )
    
    return assignment
```

##### 1.2.3 `step()`メソッドの変更

```python
def step(self, graph: EnvGraphsTuple, action: Action, ...) -> Tuple[...]:
    # ... 既存の処理 ...
    
    # フォーメーションモードの場合、フォロワーのゴールを更新
    if self._params.get("formation_mode", False):
        formation_offsets = self._params["formation_offsets"]
        if formation_offsets is not None:
            leader_pos = next_agent_states[0, :2]
            
            if self._params.get("formation_flexible_assignment", False):
                # 柔軟な割り当てモード
                follower_indices = jnp.arange(1, self.num_agents)
                follower_positions = next_agent_states[1:, :2]
                offset_indices = self.assign_formation_offsets(
                    leader_pos, follower_positions, formation_offsets
                )
                
                # 割り当てられたオフセットに基づいてゴールを設定
                for i, offset_idx in enumerate(follower_indices):
                    offset = formation_offsets[offset_indices[i]]
                    goal_states = goal_states.at[i + 1, :2].set(leader_pos + offset)
            else:
                # 既存の固定割り当てモード
                for i in range(1, min(len(formation_offsets) + 1, self.num_agents)):
                    offset = jnp.array(formation_offsets[i-1])
                    goal_states = goal_states.at[i, :2].set(leader_pos + offset)
    
    # ... 既存の処理続き ...
```

##### 1.2.4 `forward_graph()`メソッドの変更

`step()`と同様の変更を`forward_graph()`にも適用します。

### 2. リーダー優先の衝突回避（P2対応）

#### 2.1 概要

リーダー（エージェント0）は他のエージェントに対して衝突回避を行わず、障害物に対してのみ衝突回避を行います。フォロワーはリーダーを含むすべてのエージェントと障害物に対して衝突回避を行います。

#### 2.2 変更箇所

**ファイル**: `gcbfplus/algo/utils.py`

##### 2.2.1 `pwise_cbf_double_integrator_`関数の変更

```python
def pwise_cbf_double_integrator_(
    state: Array, 
    agent_idx: int, 
    o_obs_state: Array, 
    a_state: Array, 
    r: float, 
    k: int,
    leader_priority: bool = False,  # 新規追加: リーダー優先モード
    leader_idx: int = 0,  # 新規追加: リーダーのインデックス
):
    n_agent = len(a_state)
    
    pos = state[:2]
    all_obs_state = jnp.concatenate([a_state, o_obs_state], axis=0)
    all_obs_pos = all_obs_state[:, :2]
    
    # リーダー優先モードの場合
    if leader_priority:
        if agent_idx == leader_idx:
            # リーダーは他のエージェントを障害物として扱わない
            # エージェント部分を除外し、障害物のみを考慮
            agent_mask = jnp.arange(len(all_obs_state)) >= n_agent
            all_obs_state = jnp.where(
                agent_mask[:, None],
                all_obs_state,
                jnp.array([jnp.inf, jnp.inf, 0.0, 0.0])  # エージェントを無視
            )
            all_obs_pos = all_obs_state[:, :2]
        else:
            # フォロワーはすべてのエージェントと障害物を考慮（既存の動作）
            pass
    
    # 既存の処理続き
    o_dist_sq = ((pos - all_obs_pos) ** 2).sum(axis=-1)
    o_dist_sq = o_dist_sq.at[agent_idx].set(1e2)
    
    # リーダー優先モードでリーダーの場合、エージェントを除外
    if leader_priority and agent_idx == leader_idx:
        # エージェント部分の距離を大きくして除外
        o_dist_sq = o_dist_sq.at[:n_agent].set(1e2)
    
    k_idx = jnp.argsort(o_dist_sq)[:k]
    k_dist_sq = o_dist_sq[k_idx]
    k_dist_sq = k_dist_sq - 4 * r ** 2
    
    k_h0 = k_dist_sq
    k_xdiff = state[:2] - all_obs_state[k_idx][:, :2]
    k_vdiff = state[2:] - all_obs_state[k_idx][:, 2:]
    
    k_h0_dot = 2 * (k_xdiff * k_vdiff).sum(axis=-1)
    k_h1 = k_h0_dot + 10.0 * k_h0
    k_isobs = k_idx >= n_agent
    
    return k_h1, k_isobs
```

##### 2.2.2 `pwise_cbf_double_integrator`関数の変更

```python
def pwise_cbf_double_integrator(
    graph: GraphsTuple, 
    r: float, 
    n_agent: int, 
    n_rays: int, 
    k: int,
    leader_priority: bool = False,  # 新規追加
    leader_idx: int = 0,  # 新規追加
):
    a_states = graph.type_states(type_idx=0, n_type=n_agent)
    obs_states = graph.type_states(type_idx=2, n_type=n_agent * n_rays)
    a_obs_states = ei.rearrange(obs_states, "(n_agent n_ray) d -> n_agent n_ray d", n_agent=n_agent)
    
    agent_idx = jnp.arange(n_agent)
    fn = jax.vmap(
        ft.partial(
            pwise_cbf_double_integrator_, 
            r=r, 
            k=k,
            leader_priority=leader_priority,
            leader_idx=leader_idx,
        ), 
        in_axes=(0, 0, 0, None)
    )
    ak_h0, ak_isobs = fn(a_states, agent_idx, a_obs_states, a_states)
    return ak_h0, ak_isobs
```

##### 2.2.3 CBF関数取得の変更

**ファイル**: `gcbfplus/algo/utils.py`

```python
def get_pwise_cbf_fn(env: MultiAgentEnv, k: int, leader_priority: bool = False, leader_idx: int = 0):
    """CBF関数を取得する。フォーメーションモードの場合はリーダー優先を適用。"""
    env_name = env.__class__.__name__
    
    if env_name == "DoubleIntegrator":
        # 環境のパラメータからフォーメーションモードを確認
        formation_mode = env.params.get("formation_mode", False)
        use_leader_priority = formation_mode and leader_priority
        
        return ft.partial(
            pwise_cbf_double_integrator,
            r=env.params["car_radius"],
            n_agent=env.num_agents,
            n_rays=env.params["n_rays"],
            k=k,
            leader_priority=use_leader_priority,
            leader_idx=leader_idx,
        )
    # ... 他の環境タイプの処理 ...
```

##### 2.2.4 アルゴリズムクラスでの使用

**ファイル**: `gcbfplus/algo/gcbf_plus.py`, `gcbfplus/algo/gcbf.py`, `gcbfplus/algo/dec_share_cbf.py`

各アルゴリズムクラスの`__init__`メソッドで、環境のパラメータを確認し、リーダー優先モードを有効化します。

```python
def __init__(self, ...):
    # ... 既存の初期化処理 ...
    
    # フォーメーションモードの場合、リーダー優先を有効化
    formation_mode = env.params.get("formation_mode", False)
    leader_priority = formation_mode  # フォーメーションモード時は自動的に有効化
    
    self.cbf = get_pwise_cbf_fn(
        env, 
        self.k, 
        leader_priority=leader_priority,
        leader_idx=0,  # エージェント0がリーダー
    )
```

### 3. フォロワー目標地点の安全性考慮と動的オフセット調整（P3対応）

#### 3.1 概要

フォロワーの目標地点（リーダー位置 + オフセット）が障害物と衝突する可能性がある場合、オフセットを動的に調整します。

#### 3.2 変更箇所

**ファイル**: `gcbfplus/env/double_integrator.py`

##### 3.2.1 オフセット安全性チェック関数の追加

```python
def check_offset_safety(
    self,
    leader_pos: Array,
    offset: Array,
    obstacles: Obstacle,
    car_radius: float,
    comm_radius: float,
) -> Tuple[Array, bool]:
    """
    オフセット位置が安全かどうかをチェックする。
    
    Args:
        leader_pos: リーダーの位置 [x, y]
        offset: オフセット [x, y]
        obstacles: 障害物
        car_radius: エージェントの半径
        comm_radius: 通信半径（検出範囲）
    
    Returns:
        (adjusted_offset, is_safe): 調整されたオフセットと安全性フラグ
    """
    target_pos = leader_pos + offset
    
    # 障害物との衝突チェック
    from .utils import inside_obstacles
    is_inside = inside_obstacles(target_pos[None, :], obstacles, r=car_radius)
    is_safe = not is_inside[0]
    
    if is_safe:
        return offset, True
    
    # 安全でない場合、オフセットを調整
    # 方法1: オフセットの方向を維持しつつ、距離を縮める
    offset_norm = jnp.linalg.norm(offset)
    if offset_norm < 1e-6:
        # オフセットがほぼ0の場合、デフォルトの安全な方向を選択
        adjusted_offset = jnp.array([car_radius * 2, 0.0])
        return adjusted_offset, False
    
    offset_dir = offset / offset_norm
    
    # 障害物を回避する方向にオフセットを調整
    # 簡易実装: オフセット方向を90度回転させた方向を試す
    rotated_dir1 = jnp.array([-offset_dir[1], offset_dir[0]])
    rotated_dir2 = jnp.array([offset_dir[1], -offset_dir[0]])
    
    # 両方向を試して、安全な方を選択
    candidate1 = rotated_dir1 * offset_norm
    candidate2 = rotated_dir2 * offset_norm
    
    pos1 = leader_pos + candidate1
    pos2 = leader_pos + candidate2
    
    safe1 = not inside_obstacles(pos1[None, :], obstacles, r=car_radius)[0]
    safe2 = not inside_obstacles(pos2[None, :], obstacles, r=car_radius)[0]
    
    if safe1:
        return candidate1, True
    elif safe2:
        return candidate2, True
    else:
        # どちらも安全でない場合、距離を縮める
        min_safe_dist = car_radius * 3
        adjusted_offset = offset_dir * min_safe_dist
        return adjusted_offset, False
```

##### 3.2.2 動的オフセット調整の実装

```python
def adjust_formation_offsets(
    self,
    leader_pos: Array,
    formation_offsets: Array,
    obstacles: Obstacle,
    current_follower_positions: Array = None,
) -> Array:
    """
    障害物を考慮してフォーメーションオフセットを動的に調整する。
    
    Args:
        leader_pos: リーダーの位置 [x, y]
        formation_offsets: 元のオフセット [n_offsets, 2]
        obstacles: 障害物
        current_follower_positions: フォロワーの現在位置 [n_followers, 2] (オプション)
    
    Returns:
        調整されたオフセット [n_offsets, 2]
    """
    n_offsets = formation_offsets.shape[0]
    adjusted_offsets = jnp.zeros_like(formation_offsets)
    
    def adjust_one_offset(carry, offset_idx):
        adjusted_offsets, leader_pos, formation_offsets, obstacles = carry
        offset = formation_offsets[offset_idx]
        
        adjusted_offset, _ = self.check_offset_safety(
            leader_pos,
            offset,
            obstacles,
            self._params["car_radius"],
            self._params["comm_radius"],
        )
        
        adjusted_offsets = adjusted_offsets.at[offset_idx].set(adjusted_offset)
        return (adjusted_offsets, leader_pos, formation_offsets, obstacles), None
    
    (adjusted_offsets, _, _, _), _ = jax.lax.scan(
        adjust_one_offset,
        (adjusted_offsets, leader_pos, formation_offsets, obstacles),
        jnp.arange(n_offsets)
    )
    
    return adjusted_offsets
```

##### 3.2.3 `step()`メソッドでの動的オフセット調整の適用

```python
def step(self, graph: EnvGraphsTuple, action: Action, ...) -> Tuple[...]:
    # ... 既存の処理 ...
    
    # フォーメーションモードの場合、フォロワーのゴールを更新
    if self._params.get("formation_mode", False):
        formation_offsets = self._params["formation_offsets"]
        if formation_offsets is not None:
            leader_pos = next_agent_states[0, :2]
            obstacles = graph.env_states.obstacle
            
            # 動的オフセット調整が有効な場合
            if self._params.get("formation_dynamic_offset", False):
                # 障害物を考慮してオフセットを調整
                adjusted_offsets = self.adjust_formation_offsets(
                    leader_pos,
                    formation_offsets,
                    obstacles,
                )
                formation_offsets = adjusted_offsets
            
            # 柔軟な割り当てまたは固定割り当てでゴールを設定
            # ... (1.2.3のコードと同様) ...
    
    # ... 既存の処理続き ...
```

##### 3.2.4 PARAMSの拡張

```python
PARAMS = {
    # ... 既存のパラメータ ...
    "formation_dynamic_offset": False,  # 新規追加: 動的オフセット調整を有効化
    "formation_offset_safety_margin": 0.05,  # 新規追加: オフセット調整時の安全マージン
}
```

### 4. 統合的な実装の考慮事項

#### 4.1 パラメータの優先順位

1. **柔軟な割り当て** (`formation_flexible_assignment`): フォロワー間の交差問題を解決
2. **リーダー優先** (`formation_mode`が有効な場合、自動的に有効化): リーダーとフォロワーの優先順位問題を解決
3. **動的オフセット調整** (`formation_dynamic_offset`): 障害物によるフォーメーション維持不能問題を解決

#### 4.2 実装順序

1. **Phase 1**: リーダー優先の衝突回避（P2対応）
   - 最も重要で、他の改善の基盤となる
   - CBF関数の変更が必要

2. **Phase 2**: 柔軟なフォーメーション割り当て（P1対応）
   - リーダー優先が実装された後に実装
   - 環境クラスの変更が必要

3. **Phase 3**: 動的オフセット調整（P3対応）
   - 最後に実装
   - 環境クラスの変更が必要

#### 4.3 テストケース

各改善を実装した後、以下のテストケースで動作確認が必要です：

1. **P1テスト**: フォロワーが交差する初期配置で、柔軟な割り当てが機能するか
2. **P2テスト**: リーダーがフォロワーに近づいた際、リーダーが回避せずにフォロワーが回避するか
3. **P3テスト**: 障害物がフォーメーション幅より狭い場合、オフセットが動的に調整されるか

## 実装時の注意事項

### JAXの制約

- JAXの`at`によるインデックス更新は関数型なので、ループ内で逐次更新する必要がある
- `jax.lax.scan`や`jax.lax.fori_loop`を使用して効率的に実装する

### パフォーマンス

- 動的オフセット調整は各ステップで実行されるため、計算コストを考慮する
- 必要に応じて、オフセット調整の頻度を下げる（例: 数ステップに1回のみ）

### 後方互換性

- 既存のパラメータ（`formation_mode`, `formation_offsets`）は維持する
- 新しいパラメータはデフォルトで`False`または`None`にして、既存の動作を維持する

## 参考実装箇所

- CBF関数: `gcbfplus/algo/utils.py` (79-124行目)
- QPソルバー: `gcbfplus/algo/gcbf_plus.py` (299-352行目)
- 環境クラス: `gcbfplus/env/double_integrator.py` (19-473行目)
- フォーメーション実装: `gcbfplus/env/double_integrator.py` (112-121行目, 173-181行目, 374-381行目)

