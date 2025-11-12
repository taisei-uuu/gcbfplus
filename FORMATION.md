# フォーメーション飛行実装計画書

## 1. 目的

3つのエージェントがフォーメーションを組んで飛行するシミュレーションをGCBF+アルゴリズムで実現する。

### 1.1 目標

- **リーダー（エージェント0）**: 固定のゴールに向かって進む（既存の動作を維持）
- **フォロワー（エージェント1, 2）**: リーダーとの相対位置を動的ゴールとして維持しながら進む

### 1.2 対象環境

- **環境**: `DoubleIntegrator`（2D環境）
- **エージェント数**: 3（リーダー1 + フォロワー2）
- **状態次元**: 4次元（x, y, vx, vy）

## 2. 現状の理解

### 2.1 DoubleIntegrator環境の構造

- **状態表現**: `[x, y, vx, vy]` - 位置と速度
- **ゴール表現**: 各エージェントごとに固定のゴール位置 `[gx, gy, 0, 0]`（速度は0）
- **グラフ構造**: 
  - `type_states(type_idx=0)`: エージェント状態（`num_agents`個）
  - `type_states(type_idx=1)`: ゴール状態（`num_agents`個）
  - `type_states(type_idx=2)`: 障害物検出データ（LiDAR）

### 2.2 主要メソッド

- `reset()`: エージェントとゴールの初期位置をランダムに生成
- `step()`: エージェントの状態を更新（ゴールは固定のまま）
- `forward_graph()`: グラフを1ステップ前進（ゴールは更新されない）
- `u_ref()`: 各エージェントが自分のゴールに向かう制御入力を計算

### 2.3 現在の動作

現在、各エージェントは独立した固定ゴールに向かって移動する。フォーメーション機能は実装されていない。

## 3. 実装方針

### 3.1 基本アプローチ

1. **リーダー（エージェント0）**: 既存通り、固定ゴールに向かう
2. **フォロワー（エージェント1, 2）**: 各ステップでリーダーの現在位置に基づいて動的にゴールを更新
   - フォロワーのゴール = リーダーの位置 + 相対位置オフセット

### 3.2 フォーメーション設定

- **デフォルトフォーメーション**: 三角形（例）
  - エージェント0（リーダー）: 中心
  - エージェント1: リーダーの右側 `[offset_x, offset_y]`
  - エージェント2: リーダーの左側 `[-offset_x, offset_y]`

## 4. 変更が必要な箇所

### 4.1 `gcbfplus/env/double_integrator.py`

#### 4.1.1 PARAMSの拡張

**場所**: `PARAMS`辞書（35-42行目付近）

**変更内容**:
```python
PARAMS = {
    "car_radius": 0.05,
    "comm_radius": 0.5,
    "n_rays": 32,
    "obs_len_range": [0.1, 0.5],
    "n_obs": 8,
    "m": 0.1,
    # 以下を追加
    "formation_mode": False,  # フォーメーションモードの有効/無効
    "formation_offsets": None,  # フォロワーの相対位置オフセット [[x1, y1], [x2, y2], ...]
}
```

#### 4.1.2 `reset()`メソッドの変更

**場所**: 83-112行目

**変更内容**:
- フォーメーションモードが有効な場合、リーダー（エージェント0）のみ固定ゴールを生成
- フォロワー（エージェント1, 2）のゴールは、リーダーの初期位置に基づいて設定

**実装ロジック**:
```python
def reset(self, key: Array) -> GraphsTuple:
    # ... 既存の障害物生成コード ...
    
    # 既存のゴール生成ロジック（リーダー用）
    states, goals = get_node_goal_rng(...)
    
    # 速度を追加
    states = jnp.concatenate([states, jnp.zeros((self.num_agents, 2))], axis=1)
    goals = jnp.concatenate([goals, jnp.zeros((self.num_agents, 2))], axis=1)
    
    # フォーメーションモードの場合
    if self._params.get("formation_mode", False):
        formation_offsets = self._params["formation_offsets"]
        if formation_offsets is not None:
            leader_pos = states[0, :2]  # リーダーの位置 [x, y]
            # フォロワーのゴールを設定
            for i in range(1, min(len(formation_offsets) + 1, self.num_agents)):
                offset = jnp.array(formation_offsets[i-1])
                goals = goals.at[i, :2].set(leader_pos + offset)
    
    env_states = self.EnvState(states, goals, obstacles)
    return self.get_graph(env_states)
```

**注意点**:
- JAXの`at`によるインデックス更新は関数型なので、ループ内で逐次更新する必要がある
- または`jax.lax.fori_loop`を使用して効率的に実装可能

#### 4.1.3 `step()`メソッドの変更

**場所**: 145-181行目

**変更内容**:
- フォーメーションモードが有効な場合、各ステップでフォロワーのゴールを更新

**実装ロジック**:
```python
def step(self, graph: EnvGraphsTuple, action: Action, ...) -> Tuple[...]:
    self._t += 1
    
    # 既存の処理
    agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
    goal_states = graph.type_states(type_idx=1, n_type=self.num_agents)
    obstacles = graph.env_states.obstacle
    action = self.clip_action(action)
    
    next_agent_states = self.agent_step_euler(agent_states, action)
    
    # フォーメーションモードの場合、フォロワーのゴールを更新
    if self._params.get("formation_mode", False):
        formation_offsets = self._params["formation_offsets"]
        if formation_offsets is not None:
            leader_pos = next_agent_states[0, :2]  # リーダーの新しい位置
            # フォロワーのゴールを更新
            for i in range(1, min(len(formation_offsets) + 1, self.num_agents)):
                offset = jnp.array(formation_offsets[i-1])
                goal_states = goal_states.at[i, :2].set(leader_pos + offset)
    
    # 既存の処理続き
    done = jnp.array(False)
    reward = jnp.zeros(()).astype(jnp.float32)
    reward -= (jnp.linalg.norm(action - self.u_ref(graph), axis=1) ** 2).mean()
    cost = self.get_cost(graph)
    
    next_state = self.EnvState(next_agent_states, goal_states, obstacles)
    return self.get_graph(next_state), reward, cost, done, info
```

#### 4.1.4 `forward_graph()`メソッドの変更

**場所**: 340-354行目

**変更内容**:
- `step()`と同様に、フォロワーのゴールを更新

**実装ロジック**:
```python
def forward_graph(self, graph: GraphsTuple, action: Action) -> GraphsTuple:
    agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
    goal_states = graph.type_states(type_idx=1, n_type=self.num_agents)
    obs_states = graph.type_states(type_idx=2, n_type=self._params["n_rays"] * self.num_agents)
    action = self.clip_action(action)
    
    next_agent_states = self.agent_step_euler(agent_states, action)
    
    # フォーメーションモードの場合、フォロワーのゴールを更新
    if self._params.get("formation_mode", False):
        formation_offsets = self._params["formation_offsets"]
        if formation_offsets is not None:
            leader_pos = next_agent_states[0, :2]
            for i in range(1, min(len(formation_offsets) + 1, self.num_agents)):
                offset = jnp.array(formation_offsets[i-1])
                goal_states = goal_states.at[i, :2].set(leader_pos + offset)
    
    next_states = jnp.concatenate([next_agent_states, goal_states, obs_states], axis=0)
    next_graph = self.add_edge_feats(graph, next_states)
    return next_graph
```

### 4.2 `gcbfplus/env/__init__.py`

#### 4.2.1 `make_env()`関数の変更

**場所**: 23-46行目

**変更内容**:
- フォーメーションパラメータを受け取れるように引数を追加

**実装ロジック**:
```python
def make_env(
        env_id: str,
        num_agents: int,
        area_size: float = None,
        max_step: int = None,
        max_travel: Optional[float] = None,
        num_obs: Optional[int] = None,
        n_rays: Optional[int] = None,
        formation_mode: bool = False,  # 追加
        formation_offsets: Optional[list] = None,  # 追加
) -> MultiAgentEnv:
    assert env_id in ENV.keys(), f'Environment {env_id} not implemented.'
    params = ENV[env_id].PARAMS.copy()  # copy()を追加して元の辞書を変更しないように
    max_step = DEFAULT_MAX_STEP if max_step is None else max_step
    if num_obs is not None:
        params['n_obs'] = num_obs
    if n_rays is not None:
        params['n_rays'] = n_rays
    
    # フォーメーションパラメータを追加
    if formation_mode:
        params["formation_mode"] = True
        if formation_offsets is None:
            # デフォルトのオフセット（三角形フォーメーション）
            # エージェント1: 右側、エージェント2: 左側
            formation_offsets = [[0.3, 0.0], [-0.3, 0.0]]
        # JAX配列に変換して保存
        params["formation_offsets"] = jnp.array(formation_offsets)
    else:
        params["formation_mode"] = False
        params["formation_offsets"] = None
    
    return ENV[env_id](
        num_agents=num_agents,
        area_size=area_size,
        max_step=max_step,
        max_travel=max_travel,
        dt=0.03,
        params=params
    )
```

**注意点**:
- `jnp.array`をインポートする必要がある（または`jax.numpy as jnp`）

### 4.3 `train.py`

#### 4.3.1 コマンドライン引数の追加

**場所**: `main()`関数内、`argparse`の設定部分（115-150行目付近）

**変更内容**:
```python
def main():
    parser = argparse.ArgumentParser()
    # ... 既存の引数 ...
    
    # フォーメーション関連の引数追加
    parser.add_argument(
        "--formation-mode",
        action="store_true",
        default=False,
        help="Enable formation mode (leader-follower formation)"
    )
    parser.add_argument(
        "--formation-offsets",
        type=str,
        default=None,
        help="Formation offsets as JSON string, e.g., '[[0.3,0.0],[-0.3,0.0]]'"
    )
    
    args = parser.parse_args()
    
    # formation_offsetsのパース
    formation_offsets = None
    if args.formation_offsets:
        import json
        formation_offsets = json.loads(args.formation_offsets)
    
    # 環境作成時にフォーメーションパラメータを渡す
    env = make_env(
        env_id=args.env,
        num_agents=args.num_agents,
        num_obs=args.obs,
        n_rays=args.n_rays,
        area_size=args.area_size,
        formation_mode=args.formation_mode,  # 追加
        formation_offsets=formation_offsets,  # 追加
    )
    env_test = make_env(
        env_id=args.env,
        num_agents=args.num_agents,
        num_obs=args.obs,
        n_rays=args.n_rays,
        area_size=args.area_size,
        formation_mode=args.formation_mode,  # 追加
        formation_offsets=formation_offsets,  # 追加
    )
    
    # ... 既存の処理続き ...
```

### 4.4 `test.py`

#### 4.4.1 コマンドライン引数の追加

**場所**: `main()`関数内、`argparse`の設定部分（239-263行目付近）

**変更内容**:
```python
def main():
    parser = argparse.ArgumentParser()
    # ... 既存の引数 ...
    
    # フォーメーション関連の引数追加
    parser.add_argument(
        "--formation-mode",
        action="store_true",
        default=False,
        help="Enable formation mode (leader-follower formation)"
    )
    parser.add_argument(
        "--formation-offsets",
        type=str,
        default=None,
        help="Formation offsets as JSON string, e.g., '[[0.3,0.0],[-0.3,0.0]]'"
    )
    
    args = parser.parse_args()
    
    # formation_offsetsのパース
    formation_offsets = None
    if args.formation_offsets:
        import json
        formation_offsets = json.loads(args.formation_offsets)
    
    # 環境作成時にフォーメーションパラメータを渡す
    env = make_env(
        env_id=config.env if args.env is None else args.env,
        num_agents=num_agents,
        num_obs=args.obs,
        area_size=args.area_size,
        max_step=args.max_step,
        max_travel=args.max_travel,
        formation_mode=args.formation_mode,  # 追加
        formation_offsets=formation_offsets,  # 追加
    )
    
    # ... 既存の処理続き ...
```

## 5. 実装手順

1. **環境クラスのパラメータ追加**
   - `DoubleIntegrator.PARAMS`に`formation_mode`と`formation_offsets`を追加

2. **`make_env`関数の変更**
   - フォーメーションパラメータを受け取る引数を追加
   - パラメータを環境に渡す処理を追加

3. **`reset`メソッドの変更**
   - フォーメーションモード時、フォロワーのゴールをリーダー位置ベースで初期化

4. **`step`メソッドの変更**
   - フォーメーションモード時、各ステップでフォロワーのゴールを更新

5. **`forward_graph`メソッドの変更**
   - フォーメーションモード時、フォロワーのゴールを更新

6. **`train.py`の変更**
   - コマンドライン引数を追加
   - 環境作成時にパラメータを渡す

7. **`test.py`の変更**
   - コマンドライン引数を追加
   - 環境作成時にパラメータを渡す

8. **動作確認**
   - 3エージェントでフォーメーションモードを有効化してテスト

## 6. 使用例

### 6.1 訓練時

```bash
python train.py \
    --algo gcbf+ \
    --env DoubleIntegrator \
    -n 3 \
    --area-size 4 \
    --formation-mode \
    --formation-offsets '[[0.3,0.0],[-0.3,0.0]]' \
    --steps 1000 \
    --loss-action-coef 1e-4 \
    --n-env-train 16 \
    --lr-actor 1e-5 \
    --lr-cbf 1e-5 \
    --horizon 32
```

### 6.2 テスト時

```bash
python test.py \
    --path <path-to-log> \
    --epi 5 \
    --area-size 4 \
    -n 3 \
    --formation-mode \
    --formation-offsets '[[0.3,0.0],[-0.3,0.0]]'
```

### 6.3 異なるフォーメーションの例

- **横一列**: `'[[0.3,0.0],[-0.3,0.0]]'`
- **縦一列**: `'[[0.0,0.3],[0.0,-0.3]]'`
- **三角形**: `'[[0.3,0.2],[-0.3,0.2]]'`
- **V字**: `'[[0.3,-0.2],[-0.3,-0.2]]'`

## 7. 注意事項

### 7.1 JAXの制約

- JAXの`at`によるインデックス更新は関数型なので、ループ内で逐次更新する必要がある
- より効率的な実装には`jax.lax.fori_loop`を使用可能

### 7.2 エージェント数の制約

- フォロワー数は`formation_offsets`の長さに依存
- `num_agents - 1 <= len(formation_offsets)`である必要がある
- エージェント数が3より多い場合、余分なエージェントは既存の動作（独立ゴール）を維持

### 7.3 ゴール更新のタイミング

- `step()`と`forward_graph()`の両方でゴールを更新する必要がある
- `step()`は実際のシミュレーション用
- `forward_graph()`はグラフの前進計算用（訓練時など）

### 7.4 状態次元

- DoubleIntegratorは2D環境なので、位置は`[:2]`でアクセス
- 速度成分は`[2:]`でアクセス
- ゴールの速度成分は常に0なので、位置のみ更新

### 7.5 フォーメーションオフセットの単位

- オフセットは環境の座標系と同じ単位（例: メートル）
- `comm_radius`（デフォルト0.5）を考慮して、オフセットを設定することを推奨
- オフセットが大きすぎると、フォロワーがリーダーに追従できなくなる可能性がある

## 8. 期待される動作

### 8.1 リーダー（エージェント0）

- 固定のゴールに向かって移動
- GCBF+アルゴリズムにより、障害物を回避しながらゴールに向かう

### 8.2 フォロワー（エージェント1, 2）

- リーダーの現在位置に基づいて動的にゴールが更新される
- リーダーとの相対位置を維持しながら移動
- リーダーが障害物を回避する際、フォロワーも自動的に回避軌道を取る

### 8.3 フォーメーション

- 3つのエージェントが指定された相対位置を維持しながら移動
- リーダーがゴールに到達すると、フォロワーもリーダー周辺の指定位置に到達

## 9. 今後の拡張可能性

1. **動的フォーメーション変更**: 実行中にフォーメーション形状を変更
2. **複数リーダー**: 複数のリーダーと複数のフォロワーグループ
3. **階層フォーメーション**: フォロワーがさらにフォロワーを持つ構造
4. **フォーメーション学習**: 最適なフォーメーション形状を学習

## 10. 参考

- DoubleIntegrator環境: `gcbfplus/env/double_integrator.py`
- 環境ファクトリー: `gcbfplus/env/__init__.py`
- 訓練スクリプト: `train.py`
- テストスクリプト: `test.py`
