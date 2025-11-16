import numpy as np
from typing import Optional


class ComputationalGraphRecorder:
    """計算グラフを記録するクラス"""

    def __init__(self):
        self.nodes: list[dict] = []
        self.edges: list[dict] = []
        self.node_counter = 0
        self.enabled = False

    def enable(self):
        """計算グラフの記録を有効化"""
        self.enabled = True
        self.nodes = []
        self.edges = []
        self.node_counter = 0

    def disable(self):
        """計算グラフの記録を無効化"""
        self.enabled = False

    def add_node(
        self,
        name: str,
        shape: Optional[tuple] = None,
        node_type: str = "operation",
    ) -> int:
        """ノードを追加

        Args:
            name: ノード名（層の名前や操作名）
            shape: テンソルの形状
            node_type: ノードのタイプ（operation, data, parameter）

        Returns:
            ノードID
        """
        if not self.enabled:
            return -1

        node_id = self.node_counter
        self.node_counter += 1

        self.nodes.append(
            {
                "id": node_id,
                "name": name,
                "shape": shape,
                "type": node_type,
            }
        )

        return node_id

    def add_edge(
        self,
        from_node: int,
        to_node: int,
        label: str = "",
        edge_type: str = "forward",
    ):
        """エッジを追加

        Args:
            from_node: 始点ノードID
            to_node: 終点ノードID
            label: エッジのラベル
            edge_type: エッジのタイプ（forward, backward）
        """
        if not self.enabled:
            return

        self.edges.append(
            {
                "from": from_node,
                "to": to_node,
                "label": label,
                "type": edge_type,
            }
        )

    def to_dot(self, direction: str = "LR") -> str:
        """DOT言語形式で出力

        Args:
            direction: グラフの方向（LR: 左→右, TB: 上→下）

        Returns:
            DOT言語の文字列
        """
        dot = ["digraph ComputationalGraph {"]
        dot.append(f"    rankdir={direction};")
        dot.append("    node [shape=box, style=rounded];")
        dot.append("")

        # ノードの定義
        for node in self.nodes:
            node_id = node["id"]
            name = node["name"]
            shape = node["shape"]
            node_type = node["type"]

            # ノードタイプによって色を変える
            if node_type == "data":
                color = "lightblue"
            elif node_type == "parameter":
                color = "lightgreen"
            else:  # operation
                color = "lightyellow"

            label = name
            if shape is not None:
                label += f"\\nshape: {shape}"

            dot.append(
                f'    node{node_id} [label="{label}", fillcolor="{color}", style="filled,rounded"];'
            )

        dot.append("")

        # エッジの定義
        for edge in self.edges:
            from_node = edge["from"]
            to_node = edge["to"]
            label = edge["label"]
            edge_type = edge["type"]

            # エッジタイプによってスタイルを変える
            if edge_type == "backward":
                style = "dashed"
                color = "red"
            else:  # forward
                style = "solid"
                color = "black"

            edge_attrs = [f'color="{color}"', f'style="{style}"']
            if label:
                edge_attrs.append(f'label="{label}"')

            attrs = ", ".join(edge_attrs)
            dot.append(f"    node{from_node} -> node{to_node} [{attrs}];")

        dot.append("}")
        return "\n".join(dot)

    def save_dot(self, filepath: str, direction: str = "LR"):
        """DOT言語ファイルとして保存

        Args:
            filepath: 保存先ファイルパス
            direction: グラフの方向（LR: 左→右, TB: 上→下）
        """
        with open(filepath, "w") as f:
            f.write(self.to_dot(direction))

    def render_to_image(
        self, output_path: str, direction: str = "LR", format: str = "png"
    ):
        """画像ファイルとして保存

        Args:
            output_path: 出力ファイルパス（拡張子なし）
            direction: グラフの方向（LR: 左→右, TB: 上→下）
            format: 出力形式（png, pdf, svg など）

        Note:
            graphvizがインストールされている必要があります
        """
        import subprocess
        import tempfile
        import os

        # 一時的にDOTファイルを保存
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".dot", delete=False
        ) as f:
            f.write(self.to_dot(direction))
            dot_file = f.name

        try:
            # dotコマンドで画像に変換
            output_file = f"{output_path}.{format}"
            subprocess.run(
                ["dot", f"-T{format}", dot_file, "-o", output_file], check=True
            )
            print(f"Computational graph saved to: {output_file}")
        finally:
            # 一時ファイルを削除
            os.unlink(dot_file)

    def clear(self):
        """記録をクリア"""
        self.nodes = []
        self.edges = []
        self.node_counter = 0


# グローバルな計算グラフレコーダー
_global_recorder = ComputationalGraphRecorder()


def get_global_recorder() -> ComputationalGraphRecorder:
    """グローバルな計算グラフレコーダーを取得"""
    return _global_recorder
