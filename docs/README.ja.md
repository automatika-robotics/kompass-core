# Kompass Core

[![English Version][en-badge]][en-url]
[![中文版本][cn-badge]][cn-url]
[![PyPI][pypi-badge]][pypi-url]
[![MIT licensed][mit-badge]][mit-url]
[![Python Version][python-badge]][python-url]

[en-badge]: https://img.shields.io/badge/Documentation-English-green.svg
[en-url]: ../README.md
[cn-badge]: https://img.shields.io/badge/文档-中文-blue.svg
[cn-url]: README.zh.md
[pypi-badge]: https://img.shields.io/pypi/v/kompass-core.svg
[pypi-url]: https://pypi.org/project/kompass-core/
[mit-badge]: https://img.shields.io/pypi/l/kompass-core.svg
[mit-url]: https://github.com/automatika-robotics/kompass-core/LICENSE
[python-badge]: https://img.shields.io/pypi/pyversions/kompass-core.svg
[python-url]: https://www.python.org/downloads/

**Kompass Core** は、ロボットナビゲーションシステムにおける経路計画、マッピング、および制御のための高性能で GPU アクセラレーション対応のライブラリです。コアアルゴリズムは C++ で実装され、Python バインディングによってシームレスに利用できます。また、[OMPL](https://ompl.kavrakilab.org/) や [FCL](https://github.com/flexible-collision-library/fcl) との外部統合も備えています。

Kompass の理念は、「驚異的な高速性」と「高信頼性」を追求することです。GPGPU 対応の並列アルゴリズムを基盤とし、ハードウェアに依存しない設計を実現しているため、Kompass Core はさまざまなベンダーの CPU または GPU 上で実行可能です。これにより、ロボットハードウェアメーカーは、ソフトウェアスタックを大きく変更することなく、計算アーキテクチャを柔軟に切り替えることが可能です。

このパッケージは、[ROS2](https://docs.ros.org/en/rolling/index.html) 上でナビゲーションスタックを構築するための [Kompass](https://github.com/automatika-robotics/kompass) と共に使用するように設計されています。詳細な使用方法については、[Kompass ドキュメント](https://automatika-robotics.github.io/kompass/) をご覧ください。

- [**インストール**](#installation) Kompass Core 🛠️
- [**パッケージ概要**](#-package-overview) を確認
- ROS2 環境でロボットに Kompass Core を導入したい場合は、[**Kompass**](https://automatika-robotics.github.io/kompass) フレームワークをご参照ください 🚀

# インストール方法

## GPU サポート付きインストール（推奨）

Ubuntu 20 以降（Jetpack を含む）の任意のマシンに GPU サポート付きで kompass-core をインストールするには、以下を実行します：

```bash
curl https://raw.githubusercontent.com/automatika-robotics/kompass-core/refs/heads/main/build_dependencies/install_gpu.sh | bash
```

このスクリプトは、[AdaptiveCPP](https://github.com/AdaptiveCpp/AdaptiveCpp) を含むすべての関連依存関係をインストールし、`kompass-core` の最新版をソースから構築します。実行前に、[スクリプト](https://github.com/automatika-robotics/kompass-core/blob/main/build_dependencies/install_gpu.sh) の内容を確認することを推奨します。


## pip によるインストール（CPU のみ）

- Ubuntu 22.04 以降では、以下のコマンドで依存パッケージをインストールします：

```bash
  sudo apt-get install libompl-dev libfcl-dev
```
- その後、以下のように kompass-core をインストールします：

```bash
pip install kompass-core
```

PyPI では、Linux x86_64 と aarch64 向けのホイールが提供されています。なお、現時点で PyPI にあるバージョンは GPU アクセラレーションには対応していません。


## インストール内容

kompass-core をインストールすると、以下の 3 つのパッケージが使用可能になります。

- `kompass_core`：2D 空間でのナビゲーションのための運動計画と制御に関するラッパーやユーティリティを含む主要な Python API
- `kompass_cpp`：経路追跡および運動制御アルゴリズムを実装した Kompass コア C++ ライブラリの Python バインディング
- `omplpy`：Open Motion Planning Library（OMPL）向けに特化した Python バインディング

# 📦 パッケージ概要

本リポジトリには以下のモジュールが含まれます：

- `kompass_cpp/` — 計画、制御、衝突判定、マッピングアルゴリズムを実装したコア C++ モジュール

- `kompass_core/` — 設定や高レベルロジックのための Python 実装およびフロントエンドクラス

## `kompass_cpp` モジュール概要

`kompass_cpp/` はマッピング、制御、軌道計画、視覚ベースのトラッキングアルゴリズムを含む C++ パッケージであり、**GPU アクセラレーション** をサポートし、`nanobind` 経由で Python バインディングが提供されています。

### 1. マッピング
- 高速な局所マッピングアルゴリズムを実装
- **GPU アクセラレーション** に対応しリアルタイム性能を実現
- 主なクラス：`LocalMapper`, `LocalMapperGPU`

### 2. 制御と軌道計画
- PID、Stanley、動的ウィンドウ法（DWA）、ビジョンガイドコントローラなど複数の制御戦略を搭載
- **GPU アクセラレーション** による軌道サンプリングとコスト評価、重みのカスタマイズが可能
- 主なクラス：`Controller`, `PID`, `Stanley`, `DWA`, `VisionDWA`, `TrajectorySampler`, `CostEvaluator`

### 3. 衝突判定とクリティカルゾーン検出
- 安全なナビゲーションを実現する衝突判定とクリティカルゾーン検出機能を提供
- CPU 実装と GPU 実装の両方に対応
- 主なクラス：`CollisionChecker`, `CriticalZoneChecker`, `CriticalZoneCheckerGPU`

### 4. ビジョンとトラッキング
- 特徴点ベースのバウンディングボックス追跡と深度検出により認識性能を強化
- 頑健な視覚ベースのナビゲーションアルゴリズムをサポート
- 主なクラス：`FeatureBasedBboxTracker`, `DepthDetector`

### 5. ユーティリティ
- 高効率なマルチスレッド処理を実現するスレッドプール
- 実行時診断用のロガー
- 線形状態空間カルマンフィルタによる状態推定
- `tk` 名前空間で提供されるスプライン補間ユーティリティ

### 6. データ型とパラメータ
- 経路、軌道、制御、速度、バウンディングボックスを表現する豊富なデータ型
- 柔軟なパラメータ調整を可能にする強型の設定クラス

### 7. Python バインディング
- `nanobind` によって構築された包括的な Python バインディングにより、Python ワークフローとシームレスに統合可能
- マッピング、制御、ビジョン、ユーティリティの主要機能を広くカバー

## `kompass_core` モジュール概要

- `kompass_core.calibration` - ロボット運動モデルのキャリブレーション、フィッティング、ロボットシミュレーション用モジュール

- `kompass_core.control` - 多様な制御戦略と設定を含む。C++ 実装の Python ラッパークラスを提供：

| アルゴリズム名                        | 説明                                       |
|---------------------------------------|------------------------------------------|
| **Stanley**                          | 高い収束性能を持つ経路追従                |
| **DWA（動的ウィンドウ法）**         | 速度空間のサンプリングと最適化             |
| **DVZ**                              | 可変ゾーンを用いたリアクティブな障害物回避 |
| **VisionRGBFollower**               | RGB 画像を用いた視覚ターゲット追従         |
| **VisionRGBDFollower**              | RGBD（深度付き）画像を用いた視覚ターゲット追従 |

- `kompass_core.datatypes` - ロボットやセンサーデータ用の標準メッセージ・データ形式

- `kompass_core.mapping` - 局所マッピングおよび占有グリッド生成。さまざまなレーザーモデルとグリッド解像度設定に対応

- `kompass_core.models` - 差動型、全方向型、アッカーマン型ロボットの運動モデルおよび運動学をサポート。ジオメトリ定義、制御制限、シミュレーション用の状態表現も提供

- `kompass_core.motion_cost` - Python 上で使用可能な軌道評価用コストモデル（衝突確率、リファレンストラッキング、動的・静的障害物対応を含む）

- `kompass_core.performance` - アルゴリズム性能評価用モジュール

- `kompass_core.py_path_tools` - 経路補間と実行ツール

- `kompass_core.simulation` - ロボット運動のシミュレーションと経路の実行可能性評価ツール

- `kompass_core.third_party` - 外部計画ライブラリおよび衝突判定ライブラリとのラッパーと統合：

    - FCL（Flexible Collision Library）

    - OMPL（Open Motion Planning Library）

- `kompass_core.utils` - 汎用ユーティリティ群

## 著作権

本配布物に含まれるコードは、特記がない限り 2024 年 Automatika Robotics に著作権があります。
Kompass Core は MIT ライセンスのもとで公開されています。詳細は [LICENSE](LICENSE) ファイルをご覧ください。

## コントリビューション

Kompass Core は [Automatika Robotics](https://automatikarobotics.com/) と [Inria](https://inria.fr/) の共同開発プロジェクトです。コミュニティからの貢献は大歓迎です。
