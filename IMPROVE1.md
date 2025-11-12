# フォーメーション飛行の改善案

## 概要

フォーメーション飛行において観察された3つの問題点と、それらを統合的に解決するための実装計画をまとめます。

**重要**: 本改善案は以下の条件でのみ適用されます：
- **環境**: `DoubleIntegrator`環境のみ
- **アルゴリズム**: `GCBFPlus`（gcbf+）アルゴリズムのみ
- **適用条件**: `formation_mode=True`が有効な場合のみ

既存のformationモードが起動していない場合、すべての変更は適用されず、既存の動作が維持されます。

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
    "formation_assignment_cooldown": 30,  # 新規追加: 割り当て変更のクールダウン期間（ステップ数）
    "formation_assignment_min_diff": 0.05,  # 新規追加: 再割り当てを検討する最小距離差
}
```

**注意**: これらのパラメータは`formation_mode=True`の場合のみ使用されます。

##### 1.2.2 フォーメーション割り当て関数の追加（チャタリング対策付き）

```python
def assign_formation_offsets(
    self, 
    leader_pos: Array, 
    follower_positions: Array, 
    formation_offsets: Array,
    previous_assignment: Optional[Array] = None,  # 前回の割り当て
    assignment_age: Optional[Array] = None,  # 各割り当ての経過ステップ数
    cooldown_steps: int = 30,  # クールダウン期間（ステップ数）
    min_distance_diff: float = 0.05,  # 再割り当てを検討する最小距離差
) -> Tuple[Array, Array]:
    """
    フォロワーの現在位置に基づいて、最適なオフセット割り当てを決定する。
    チャタリングを防ぐため、ヒステリシス（クールダウンタイム）を実装。
    
    Args:
        leader_pos: リーダーの位置 [x, y]
        follower_positions: フォロワーの現在位置 [n_followers, 2]
        formation_offsets: 利用可能なオフセット [n_offsets, 2]
        previous_assignment: 前回の割り当て [n_followers] (オプション)
        assignment_age: 各割り当ての経過ステップ数 [n_followers] (オプション)
        cooldown_steps: 再割り当てを禁止する期間（ステップ数）
        min_distance_diff: 再割り当てを検討する最小距離差
    
    Returns:
        (assignment, new_assignment_age): 割り当てと更新された経過ステップ数
    """
    n_followers = follower_positions.shape[0]
    n_offsets = formation_offsets.shape[0]
    
    # 初期化
    if previous_assignment is None:
        previous_assignment = jnp.zeros(n_followers, dtype=jnp.int32)
    if assignment_age is None:
        assignment_age = jnp.zeros(n_followers, dtype=jnp.int32)
    
    # 各フォロワーと各オフセットの組み合わせについて、目標位置を計算
    target_positions = leader_pos[None, :] + formation_offsets[:, None, :]  # [n_offsets, n_followers, 2]
    target_positions = jnp.transpose(target_positions, (1, 0, 2))  # [n_followers, n_offsets, 2]
    
    # 各フォロワーから各目標位置への距離を計算
    distances = jnp.linalg.norm(
        follower_positions[:, None, :] - target_positions, 
        axis=-1
    )  # [n_followers, n_offsets]
    
    # 新しい割り当てを計算（クールダウンを考慮しない）
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
    
    (new_assignment, _, _), _ = jax.lax.scan(
        assign_one,
        (assignment, used_offsets, distances),
        jnp.arange(n_followers)
    )
    
    # クールダウン期間中の割り当てを維持
    # 前回の割り当てからの距離差を計算
    current_distances = jnp.take_along_axis(
        distances,
        previous_assignment[:, None],
        axis=1
    ).squeeze(1)  # [n_followers]
    
    new_distances = jnp.take_along_axis(
        distances,
        new_assignment[:, None],
        axis=1
    ).squeeze(1)  # [n_followers]
    
    distance_improvement = current_distances - new_distances  # 改善量
    
    # クールダウン期間中、または改善が小さい場合は前回の割り当てを維持
    in_cooldown = assignment_age < cooldown_steps
    small_improvement = distance_improvement < min_distance_diff
    
    should_keep_previous = jnp.logical_or(in_cooldown, small_improvement)
    
    final_assignment = jnp.where(
        should_keep_previous,
        previous_assignment,
        new_assignment
    )
    
    # 割り当てが変更された場合はageをリセット、そうでなければインクリメント
    assignment_changed = final_assignment != previous_assignment
    new_assignment_age = jnp.where(
        assignment_changed,
        jnp.zeros(n_followers, dtype=jnp.int32),
        assignment_age + 1
    )
    
    return final_assignment, new_assignment_age
```

##### 1.2.3 EnvStateの拡張（割り当て状態の保持）

```python
class EnvState(NamedTuple):
    agent: AgentState
    goal: State
    obstacle: Obstacle
    # 新規追加: フォーメーション割り当ての状態
    formation_assignment: Optional[Array] = None  # 現在の割り当て [n_followers]
    formation_assignment_age: Optional[Array] = None  # 割り当ての経過ステップ数 [n_followers]
    
    @property
    def n_agent(self) -> int:
        return self.agent.shape[0]
```

##### 1.2.4 `step()`メソッドの変更

```python
def step(self, graph: EnvGraphsTuple, action: Action, ...) -> Tuple[...]:
    # ... 既存の処理 ...
    
    # フォーメーションモードの場合、フォロワーのゴールを更新
    if self._params.get("formation_mode", False):
        formation_offsets = self._params["formation_offsets"]
        if formation_offsets is not None:
            leader_pos = next_agent_states[0, :2]
            
            if self._params.get("formation_flexible_assignment", False):
                # 柔軟な割り当てモード（チャタリング対策付き）
                follower_indices = jnp.arange(1, self.num_agents)
                follower_positions = next_agent_states[1:, :2]
                
                # 前回の割り当て状態を取得
                prev_assignment = graph.env_states.formation_assignment
                prev_age = graph.env_states.formation_assignment_age
                
                offset_indices, new_age = self.assign_formation_offsets(
                    leader_pos,
                    follower_positions,
                    formation_offsets,
                    previous_assignment=prev_assignment,
                    assignment_age=prev_age,
                    cooldown_steps=self._params.get("formation_assignment_cooldown", 30),
                    min_distance_diff=self._params.get("formation_assignment_min_diff", 0.05),
                )
                
                # 割り当てられたオフセットに基づいてゴールを設定
                for i in range(len(follower_indices)):
                    offset = formation_offsets[offset_indices[i]]
                    goal_states = goal_states.at[i + 1, :2].set(leader_pos + offset)
                
                # 割り当て状態を更新（次回のstepで使用）
                # 注意: EnvStateの更新は後で行う
            else:
                # 既存の固定割り当てモード
                for i in range(1, min(len(formation_offsets) + 1, self.num_agents)):
                    offset = jnp.array(formation_offsets[i-1])
                    goal_states = goal_states.at[i, :2].set(leader_pos + offset)
    
    # EnvStateの更新時に割り当て状態も含める
    next_state = self.EnvState(
        next_agent_states,
        goal_states,
        obstacles,
        formation_assignment=offset_indices if self._params.get("formation_flexible_assignment", False) else None,
        formation_assignment_age=new_age if self._params.get("formation_flexible_assignment", False) else None,
    )
    
    # ... 既存の処理続き ...
```

##### 1.2.5 `forward_graph()`メソッドの変更

`step()`と同様の変更を`forward_graph()`にも適用します。ただし、`forward_graph()`はグラフの前進計算のみを行うため、割り当て状態の更新は行わない（`step()`でのみ更新）。

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

**ファイル**: `gcbfplus/algo/gcbf_plus.py` **のみ**

**重要**: リーダー優先モードは`GCBFPlus`クラスのみに適用されます。他のアルゴリズム（`GCBF`, `DecShareCBF`, `CentralizedCBF`）には適用しません。

```python
def __init__(self, ...):
    # ... 既存の初期化処理 ...
    
    # DoubleIntegrator環境かつフォーメーションモードの場合のみ、リーダー優先を有効化
    env_name = env.__class__.__name__
    formation_mode = env.params.get("formation_mode", False)
    
    if env_name == "DoubleIntegrator" and formation_mode:
        leader_priority = True
    else:
        leader_priority = False
    
    self.cbf = get_pwise_cbf_fn(
        env, 
        self.k, 
        leader_priority=leader_priority,
        leader_idx=0,  # エージェント0がリーダー
    )
```
#### 2.3 L_CBF（CBF損失）計算へのマスク適用

注意: ここまでの変更は「CBF 関数の出力」および「学習時の教師データ」にリーダー優先のロジックを追加する内容でしたが、学習・最適化で実際に使われる損失項（特に gcbfplus/algo/gcbf_plus.py 内で計算される L_CBF）にも同等のマスクを入れる必要があります。さもないと、損失計算の段階でリーダーに対する他エージェント起因の h1 項が依然としてペナルティとなり、学習や制御挙動に意図しない影響を与えます。

推奨実装（gcbfplus/algo/gcbf_plus.py 内の損失計算部に挿入するコード例）:

```python
# 例: gcbfplus/algo/gcbf_plus.py の該当箇所（損失計算部）
# ak_h1: [n_agent, k] --- CBF の h1 値（agent/obstacle 混在）
# ak_isobs: [n_agent, k] --- 各制約が障害物かどうかの bool
# leader_priority: bool, leader_idx: int

# マスクを作成: 障害物制約は常に有効、エージェント由来の制約は
# リーダーに対して無視する（ゼロ化）ようにする
if leader_priority:
    n_agent = ak_h1.shape[0]
    agent_indices = jnp.arange(n_agent)  # [0..n_agent-1]
    # entry_keep: True ならその項を損失に含める
    entry_keep = ak_isobs | (agent_indices[:, None] != leader_idx)
    # マスクしてリーダーのエージェント関連制約を無視
    masked_ak_h1 = jnp.where(entry_keep, ak_h1, 0.0)
else:
    masked_ak_h1 = ak_h1

# あとは masked_ak_h1 を使って L_CBF を計算する
# 例: ReLU 等を適用してから平均あるいは合計を損失に加える
l_cbf_per_entry = jnp.maximum(0.0, -masked_ak_h1)  # 違反分だけをペナルティ
L_CBF = jnp.mean(l_cbf_per_entry)  # または適切な重み付き和

    return total_loss, {
        'loss/action': loss_action,
        'loss/unsafe': loss_unsafe,
        'loss/safe': loss_safe,
        'loss/h_dot': loss_h_dot,
        'loss/cbf': loss_cbf, #ここが手作りcbf(リーダが盲目)の評価項
        'loss/total': total_loss,
        'acc/unsafe': acc_unsafe,
        'acc/safe': acc_safe,
        'acc/h_dot': acc_h_dot,
        'acc/unsafe_data_ratio': unsafe_data_ratio
    }
```


### 3. フォロワー目標地点の安全性考慮と動的オフセット調整（P3対応）

#### 3.1 概要

フォロワーの目標地点（リーダー位置 + オフセット）が障害物と衝突する可能性がある場合、オフセットを動的に調整します。

#### 3.2 変更箇所

**ファイル**: `gcbfplus/env/double_integrator.py`

##### 3.2.1 オフセット安全性チェック関数の追加（リーダー真後ろ方式）

```python
def check_offset_safety(
    self,
    leader_pos: Array,
    leader_velocity: Array,  # リーダーの速度 [vx, vy]
    offset: Array,
    obstacles: Obstacle,
    car_radius: float,
    comm_radius: float,
    follower_idx: int,  # フォロワーのインデックス（重複回避用）
    other_follower_offsets: Optional[Array] = None,  # 他のフォロワーのオフセット
) -> Tuple[Array, bool]:
    """
    オフセット位置が安全かどうかをチェックし、安全でない場合はリーダーの真後ろに調整する。
    
    Args:
        leader_pos: リーダーの位置 [x, y]
        leader_velocity: リーダーの速度 [vx, vy]
        offset: オフセット [x, y]
        obstacles: 障害物
        car_radius: エージェントの半径
        comm_radius: 通信半径（検出範囲）
        follower_idx: フォロワーのインデックス（0-indexed、フォロワー1なら0、フォロワー2なら1）
        other_follower_offsets: 他のフォロワーの現在のオフセット [n_other_followers, 2]
    
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
    
    # 安全でない場合、リーダーの真後ろの直線上に目標地点を設定
    # リーダーの速度方向を取得（後ろ方向）
    vel_norm = jnp.linalg.norm(leader_velocity)
    
    if vel_norm < 1e-6:
        # リーダーが停止している場合、デフォルトの後ろ方向（-y方向）を使用
        backward_dir = jnp.array([0.0, -1.0])
    else:
        # リーダーの進行方向の逆（後ろ方向）
        backward_dir = -leader_velocity / vel_norm
    
    # リーダーから後ろ方向への距離を設定（複数のフォロワーが重ならないように）
    base_distance = car_radius * 3  # 基本距離
    follower_spacing = car_radius * 4  # フォロワー間の間隔
    
    # 他のフォロワーが既に真後ろを使用しているかチェック
    if other_follower_offsets is not None:
        # 他のフォロワーのオフセットが後ろ方向かチェック
        # 簡易実装: 他のフォロワーのオフセットが後ろ方向に近い場合、距離を調整
        other_backward_distances = []
        for other_offset in other_follower_offsets:
            other_offset_norm = jnp.linalg.norm(other_offset)
            if other_offset_norm > 1e-6:
                other_dir = other_offset / other_offset_norm
                # 後ろ方向との内積が大きい（後ろ方向に近い）場合
                dot_product = jnp.dot(other_dir, backward_dir)
                if dot_product > 0.7:  # 約45度以内
                    other_backward_distances.append(other_offset_norm)
        
        if len(other_backward_distances) > 0:
            # 既存のフォロワーの距離より遠くに配置
            max_other_distance = jnp.max(jnp.array(other_backward_distances))
            safe_distance = max_other_distance + follower_spacing
        else:
            safe_distance = base_distance
    else:
        safe_distance = base_distance + follower_idx * follower_spacing
    
    # リーダーの真後ろに目標地点を設定
    adjusted_offset = backward_dir * safe_distance
    adjusted_target_pos = leader_pos + adjusted_offset
    
    # 調整後の位置が安全かチェック
    adjusted_is_safe = not inside_obstacles(adjusted_target_pos[None, :], obstacles, r=car_radius)[0]
    
    return adjusted_offset, adjusted_is_safe
```

##### 3.2.2 動的オフセット調整の実装（チャタリング対策付き）

```python
def adjust_formation_offsets(
    self,
    leader_pos: Array,
    leader_velocity: Array,  # リーダーの速度 [vx, vy]
    formation_offsets: Array,
    obstacles: Obstacle,
    previous_adjusted_offsets: Optional[Array] = None,  # 前回の調整済みオフセット
    offset_state_age: Optional[Array] = None,  # 各オフセットの状態の経過ステップ数
    cooldown_steps: int = 20,  # オフセット変更のクールダウン期間
    safety_margin: float = 0.05,  # 安全性判定のマージン
) -> Tuple[Array, Array]:
    """
    障害物を考慮してフォーメーションオフセットを動的に調整する。
    チャタリングを防ぐため、ヒステリシスを実装。
    
    Args:
        leader_pos: リーダーの位置 [x, y]
        leader_velocity: リーダーの速度 [vx, vy]
        formation_offsets: 元のオフセット [n_offsets, 2]
        obstacles: 障害物
        previous_adjusted_offsets: 前回の調整済みオフセット [n_offsets, 2] (オプション)
        offset_state_age: 各オフセットの状態の経過ステップ数 [n_offsets] (オプション)
        cooldown_steps: オフセット変更のクールダウン期間（ステップ数）
        safety_margin: 安全性判定のマージン
    
    Returns:
        (adjusted_offsets, new_offset_state_age): 調整されたオフセットと更新された経過ステップ数
    """
    n_offsets = formation_offsets.shape[0]
    
    # 初期化
    if previous_adjusted_offsets is None:
        previous_adjusted_offsets = formation_offsets
    if offset_state_age is None:
        offset_state_age = jnp.zeros(n_offsets, dtype=jnp.int32)
    
    adjusted_offsets = jnp.zeros_like(formation_offsets)
    
    def adjust_one_offset(carry, offset_idx):
        adjusted_offsets, leader_pos, leader_velocity, formation_offsets, obstacles, previous_adjusted_offsets, offset_state_age = carry
        offset = formation_offsets[offset_idx]
        prev_offset = previous_adjusted_offsets[offset_idx]
        age = offset_state_age[offset_idx]
        
        # 他のフォロワーのオフセットを取得（重複回避用）
        other_offsets = jnp.concatenate([
            adjusted_offsets[:offset_idx],
            adjusted_offsets[offset_idx+1:] if offset_idx < n_offsets - 1 else jnp.array([]).reshape(0, 2)
        ], axis=0) if offset_idx > 0 or offset_idx < n_offsets - 1 else jnp.array([]).reshape(0, 2)
        
        # 新しいオフセットを計算
        new_offset, is_safe = self.check_offset_safety(
            leader_pos,
            leader_velocity,
            offset,
            obstacles,
            self._params["car_radius"],
            self._params["comm_radius"],
            follower_idx=offset_idx,
            other_follower_offsets=other_offsets if other_offsets.shape[0] > 0 else None,
        )
        
        # 前回のオフセットが安全かチェック
        prev_target_pos = leader_pos + prev_offset
        from .utils import inside_obstacles
        prev_is_safe = not inside_obstacles(prev_target_pos[None, :], obstacles, r=self._params["car_radius"])[0]
        
        # クールダウン期間中、または前回が安全で新しいものも安全な場合は前回を維持
        in_cooldown = age < cooldown_steps
        both_safe = prev_is_safe and is_safe
        
        # 前回が危険で新しいものが安全な場合は必ず変更
        # 前回が安全で新しいものも安全な場合、クールダウン中は前回を維持
        should_keep_previous = jnp.logical_and(
            in_cooldown,
            jnp.logical_or(both_safe, prev_is_safe)
        )
        
        final_offset = jnp.where(should_keep_previous, prev_offset, new_offset)
        offset_changed = jnp.any(jnp.abs(final_offset - prev_offset) > 1e-6)
        
        adjusted_offsets = adjusted_offsets.at[offset_idx].set(final_offset)
        
        # 状態の経過ステップ数を更新
        new_age = jnp.where(offset_changed, jnp.array(0, dtype=jnp.int32), age + 1)
        offset_state_age = offset_state_age.at[offset_idx].set(new_age)
        
        return (adjusted_offsets, leader_pos, leader_velocity, formation_offsets, obstacles, previous_adjusted_offsets, offset_state_age), None
    
    (adjusted_offsets, _, _, _, _, _, new_offset_state_age), _ = jax.lax.scan(
        adjust_one_offset,
        (adjusted_offsets, leader_pos, leader_velocity, formation_offsets, obstacles, previous_adjusted_offsets, offset_state_age),
        jnp.arange(n_offsets)
    )
    
    return adjusted_offsets, new_offset_state_age
```

##### 3.2.3 EnvStateの拡張（オフセット状態の保持）

```python
class EnvState(NamedTuple):
    agent: AgentState
    goal: State
    obstacle: Obstacle
    # 新規追加: フォーメーション割り当ての状態
    formation_assignment: Optional[Array] = None
    formation_assignment_age: Optional[Array] = None
    # 新規追加: 動的オフセット調整の状態
    formation_adjusted_offsets: Optional[Array] = None  # 調整済みオフセット [n_offsets, 2]
    formation_offset_state_age: Optional[Array] = None  # 各オフセットの状態の経過ステップ数 [n_offsets]
    
    @property
    def n_agent(self) -> int:
        return self.agent.shape[0]
```

##### 3.2.4 `step()`メソッドでの動的オフセット調整の適用

```python
def step(self, graph: EnvGraphsTuple, action: Action, ...) -> Tuple[...]:
    # ... 既存の処理 ...
    
    # フォーメーションモードの場合、フォロワーのゴールを更新
    if self._params.get("formation_mode", False):
        formation_offsets = self._params["formation_offsets"]
        if formation_offsets is not None:
            leader_pos = next_agent_states[0, :2]
            leader_velocity = next_agent_states[0, 2:]  # リーダーの速度
            obstacles = graph.env_states.obstacle
            
            # 動的オフセット調整が有効な場合
            if self._params.get("formation_dynamic_offset", False):
                # 前回の調整済みオフセットと状態を取得
                prev_adjusted_offsets = graph.env_states.formation_adjusted_offsets
                prev_offset_state_age = graph.env_states.formation_offset_state_age
                
                # 障害物を考慮してオフセットを調整（チャタリング対策付き）
                adjusted_offsets, new_offset_state_age = self.adjust_formation_offsets(
                    leader_pos,
                    leader_velocity,
                    formation_offsets,
                    obstacles,
                    previous_adjusted_offsets=prev_adjusted_offsets,
                    offset_state_age=prev_offset_state_age,
                    cooldown_steps=self._params.get("formation_offset_cooldown", 20),
                    safety_margin=self._params.get("formation_offset_safety_margin", 0.05),
                )
                formation_offsets = adjusted_offsets
            else:
                new_offset_state_age = None
            
            # 柔軟な割り当てまたは固定割り当てでゴールを設定
            # ... (1.2.4のコードと同様) ...
    
    # EnvStateの更新時にオフセット状態も含める
    next_state = self.EnvState(
        next_agent_states,
        goal_states,
        obstacles,
        formation_assignment=offset_indices if self._params.get("formation_flexible_assignment", False) else None,
        formation_assignment_age=new_age if self._params.get("formation_flexible_assignment", False) else None,
        formation_adjusted_offsets=adjusted_offsets if self._params.get("formation_dynamic_offset", False) else None,
        formation_offset_state_age=new_offset_state_age if self._params.get("formation_dynamic_offset", False) else None,
    )
    
    # ... 既存の処理続き ...
```

##### 3.2.5 PARAMSの拡張

```python
PARAMS = {
    # ... 既存のパラメータ ...
    "formation_dynamic_offset": False,  # 新規追加: 動的オフセット調整を有効化
    "formation_offset_safety_margin": 0.05,  # 新規追加: オフセット調整時の安全マージン
    "formation_offset_cooldown": 20,  # 新規追加: オフセット変更のクールダウン期間（ステップ数）
    "formation_follower_spacing": 0.2,  # 新規追加: 真後ろに配置する際のフォロワー間隔
}
```

**注意**: これらのパラメータは`formation_mode=True`かつ`formation_dynamic_offset=True`の場合のみ使用されます。

### 4. 統合的な実装の考慮事項

#### 4.1 適用範囲の明確化

**重要**: すべての改善は以下の条件でのみ適用されます：

1. **環境**: `DoubleIntegrator`環境のみ
2. **アルゴリズム**: `GCBFPlus`（gcbf+）アルゴリズムのみ
3. **適用条件**: `formation_mode=True`が有効な場合のみ

既存のformationモードが起動していない場合、すべての変更は適用されず、既存の動作が維持されます。

#### 4.2 パラメータの優先順位

1. **柔軟な割り当て** (`formation_flexible_assignment`): フォロワー間の交差問題を解決（チャタリング対策付き）
2. **リーダー優先** (`formation_mode`が有効な場合、`GCBFPlus`で自動的に有効化): リーダーとフォロワーの優先順位問題を解決
3. **動的オフセット調整** (`formation_dynamic_offset`): 障害物によるフォーメーション維持不能問題を解決（チャタリング対策付き）

#### 4.3 実装順序

1. **Phase 1**: リーダー優先の衝突回避（P2対応）
   - 最も重要で、他の改善の基盤となる
   - CBF関数の変更が必要

2. **Phase 2**: 柔軟なフォーメーション割り当て（P1対応）
   - リーダー優先が実装された後に実装
   - 環境クラスの変更が必要

3. **Phase 3**: 動的オフセット調整（P3対応）
   - 最後に実装
   - 環境クラスの変更が必要

#### 4.4 チャタリング対策の検証

各改善にはチャタリング対策が含まれています。以下のテストケースで動作確認が必要です：

1. **P1テスト（チャタリング）**: フォロワーが2つのオフセットに対してほぼ等距離になった場合、割り当てが頻繁に切り替わらないか
2. **P3テスト（チャタリング）**: オフセット位置が安全/危険の境界付近で、オフセットが頻繁に切り替わらないか
3. **P3テスト（重複回避）**: 複数のフォロワーが同時に真後ろに配置される場合、目標地点が重ならないか

#### 4.5 テストケース

各改善を実装した後、以下のテストケースで動作確認が必要です：

1. **P1テスト**: フォロワーが交差する初期配置で、柔軟な割り当てが機能するか
2. **P2テスト**: リーダーがフォロワーに近づいた際、リーダーが回避せずにフォロワーが回避するか
3. **P3テスト**: 障害物がフォーメーション幅より狭い場合、オフセットが動的に調整されるか（真後ろ方式）
4. **P3テスト（複数フォロワー）**: 2つのフォロワーが同時に危険な場合、両方が真後ろに配置されても目標地点が重ならないか

## 実装時の注意事項

### JAXの制約

- JAXの`at`によるインデックス更新は関数型なので、ループ内で逐次更新する必要がある
- `jax.lax.scan`や`jax.lax.fori_loop`を使用して効率的に実装する

### パフォーマンス

- 動的オフセット調整は各ステップで実行されるため、計算コストを考慮する
- チャタリング対策により、実際の再計算頻度は低くなる（クールダウン期間中は前回の値を維持）
- 必要に応じて、クールダウン期間を調整してパフォーマンスと応答性のバランスを取る

### 後方互換性

- 既存のパラメータ（`formation_mode`, `formation_offsets`）は維持する
- 新しいパラメータはデフォルトで`False`または`None`にして、既存の動作を維持する
- `formation_mode=False`の場合、すべての新機能は無効化され、既存の動作が維持される
- `DoubleIntegrator`以外の環境、`GCBFPlus`以外のアルゴリズムでは、すべての変更は適用されない

## 参考実装箇所

- CBF関数: `gcbfplus/algo/utils.py` (79-124行目)
- QPソルバー: `gcbfplus/algo/gcbf_plus.py` (299-352行目)
- 環境クラス: `gcbfplus/env/double_integrator.py` (19-473行目)
- フォーメーション実装: `gcbfplus/env/double_integrator.py` (112-121行目, 173-181行目, 374-381行目)

