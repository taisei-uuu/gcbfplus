# JAX互換性エラー修正方法

## エラー内容

`train.py`を実行した際に以下のエラーが発生します：

```
AttributeError: jax.interpreters.xla.pytype_aval_mappings was deprecated in JAX v0.5.0 and removed in JAX v0.7.0. 
jax.core.pytype_aval_mappings can be used as a replacement in most cases.
```

## 原因

1. **依存ライブラリの問題**: `jaxproxqp`ライブラリが古いJAX API（`jax.interpreters.xla.pytype_aval_mappings`）を使用している
2. **インポート順序の問題**: `compat_jax_xla_shim`が適用される前に、`jaxproxqp`がインポートされ、その時点で`jax.interpreters.xla.pytype_aval_mappings`へのアクセスが発生している
3. **適用タイミングの問題**: `train.py`の先頭で`compat_jax_xla_shim.apply()`を呼んでいるが、`gcbfplus.algo`パッケージがインポートされる際に内部で`jaxproxqp`がインポートされ、その時点でエラーが発生している

## 修正方法

### 修正1: `compat_jax_xla_shim.py`の改善

現在の実装では、モジュールが既にインポートされている場合に対応できていません。より確実に動作するように修正します。

**ファイル**: `compat_jax_xla_shim.py`

```python
# compat_jax_xla_shim.py
def apply():
    """JAX v0.7.0以降で削除されたpytype_aval_mappingsを復元する互換性シム"""
    import importlib
    import sys
    
    try:
        # jax.coreから新しいAPIを取得
        from jax.core import pytype_aval_mappings as _pam
        
        # jax.interpreters.xlaモジュールを取得（既にインポートされていてもOK）
        if 'jax.interpreters.xla' in sys.modules:
            xla = sys.modules['jax.interpreters.xla']
        else:
            xla = importlib.import_module("jax.interpreters.xla")
        
        # 属性が存在しない、または古いバージョンの場合は新しいAPIを設定
        if not hasattr(xla, "pytype_aval_mappings"):
            setattr(xla, "pytype_aval_mappings", _pam)
        # 既に存在する場合でも、新しいAPIで上書きする（より安全）
        else:
            # 古い実装が残っている可能性があるため、新しい実装で上書き
            setattr(xla, "pytype_aval_mappings", _pam)
            
    except ImportError:
        # jax.coreが存在しない古いJAXバージョンの場合は何もしない
        pass
    except Exception as e:
        # その他のエラーは無視（互換性シムなので失敗しても致命的でない）
        import warnings
        warnings.warn(f"Failed to apply JAX compatibility shim: {e}", RuntimeWarning)
```

### 修正2: `gcbfplus/algo/__init__.py`で互換性シムを適用

`gcbfplus.algo`パッケージがインポートされる際に、確実に互換性シムを適用します。

**ファイル**: `gcbfplus/algo/__init__.py`

```python
# 他のインポートより前に互換性シムを適用
import sys
import os

# プロジェクトルートのパスを追加（compat_jax_xla_shimをインポートするため）
if __name__ != "__main__":
    # パッケージのルートディレクトリを取得
    _pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _pkg_root not in sys.path:
        sys.path.insert(0, _pkg_root)

try:
    import compat_jax_xla_shim
    compat_jax_xla_shim.apply()
except ImportError:
    # compat_jax_xla_shimが見つからない場合は警告のみ
    import warnings
    warnings.warn("compat_jax_xla_shim not found. JAX compatibility may fail.", RuntimeWarning)

from .base import MultiAgentController
from .dec_share_cbf import DecShareCBF
from .gcbf import GCBF
from .gcbf_plus import GCBFPlus
from .centralized_cbf import CentralizedCBF


def make_algo(algo: str, **kwargs) -> MultiAgentController:
    if algo == 'gcbf':
        return GCBF(**kwargs)
    elif algo == 'gcbf+':
        return GCBFPlus(**kwargs)
    elif algo == 'centralized_cbf':
        return CentralizedCBF(**kwargs)
    elif algo == 'dec_share_cbf':
        return DecShareCBF(**kwargs)
    else:
        raise ValueError(f'Unknown algorithm: {algo}')
```

### 修正3: `jaxproxqp`をインポートする各ファイルで事前に適用（代替案）

より確実な方法として、`jaxproxqp`をインポートする各ファイル（`gcbf_plus.py`、`centralized_cbf.py`、`dec_share_cbf.py`）の先頭で互換性シムを適用します。

**ファイル**: `gcbfplus/algo/gcbf_plus.py`（1行目に追加）

```python
# 他のインポートより前に互換性シムを適用
import sys
import os
_pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)
try:
    import compat_jax_xla_shim
    compat_jax_xla_shim.apply()
except ImportError:
    pass

import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import optax
import jax
import functools as ft
import jax.tree_util as jtu
import numpy as np
import einops as ei

from typing import Optional, Tuple, NamedTuple
from flax.training.train_state import TrainState
from jaxproxqp.jaxproxqp import JaxProxQP  # このインポートの前に互換性シムを適用
# ... 以下既存のコード ...
```

同様に、`centralized_cbf.py`と`dec_share_cbf.py`にも同じ修正を適用します。

## 推奨される修正順序

1. **まず修正1を適用**: `compat_jax_xla_shim.py`を改善して、より確実に動作するようにする
2. **次に修正2を適用**: `gcbfplus/algo/__init__.py`で互換性シムを事前に適用する
3. **それでも解決しない場合**: 修正3を適用して、各ファイルでも個別に適用する

## 修正後の動作確認

修正後、以下のコマンドで動作確認してください：

```bash
python train.py --algo gcbf+ --env DoubleIntegrator -n 3 --area-size 4 --steps 100
```

エラーが発生しないことを確認してください。

## 補足説明

### なぜこのエラーが発生するのか

- JAX v0.7.0以降では、`jax.interpreters.xla.pytype_aval_mappings`が削除され、`jax.core.pytype_aval_mappings`に移動しました
- `jaxproxqp`は古いAPIを使用しているため、新しいJAXバージョンではエラーが発生します
- 互換性シムは、古いAPIを新しいAPIにマッピングすることで、エラーを回避します

### 代替解決策

1. **jaxproxqpをフォークして修正**: `jaxproxqp`のソースコードを修正して新しいAPIを使用する
2. **JAXのバージョンを固定**: `requirements.txt`でJAX v0.6.xを指定する（ただし、他の依存関係との互換性問題が発生する可能性がある）

互換性シムを使用する方法が最も簡単で安全です。

