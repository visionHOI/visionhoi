import torch
import collections
from collections import defaultdict


class HOIErrorAnalyzer:
    def __init__(self, unseen_ids, seen_ids):
        self.unseen_ids = set(unseen_ids)
        self.seen_ids = set(seen_ids)
        self.total_unseen = 0
        self.missed_unseen = 0
        self.bias_to_seen = 0
        self.top_seen_bias = defaultdict(int)

    def update(self, gt_hois, top_pred_id):
        # 统计 Unseen 样本的表现
        for gt_id in gt_hois:
            gt_id = gt_id.item()
            if gt_id in self.unseen_ids:
                self.total_unseen += 1
                if top_pred_id != gt_id:
                    self.missed_unseen += 1
                    # 检查是否误认为了 Seen 类
                    if top_pred_id in self.seen_ids:
                        self.bias_to_seen += 1
                        self.top_seen_bias[top_pred_id] += 1

    def report(self):
        print("\n" + "=" * 50)
        print("📊 [优化效果实时监控报告]")
        print("-" * 50)
        if self.total_unseen == 0:
            print("未检测到 Unseen 样本。")
            return

        bias_ratio = (self.bias_to_seen / self.total_unseen) * 100
        miss_ratio = (self.missed_unseen / self.total_unseen) * 100

        print(f"Unseen 样本总数: {self.total_unseen}")
        print(f"漏检/误检率: {miss_ratio:.2f}%")
        print(f"Seen 类偏见比例: {bias_ratio:.2f}% (目标: < 70%)")
        print("Top 3 干扰项 (正在被抑制):")

        sorted_bias = sorted(self.top_seen_bias.items(), key=lambda x: x[1], reverse=True)
        for i, (sid, count) in enumerate(sorted_bias[:3]):
            print(f"  - ID {sid}: {count} 次")
        print("=" * 50 + "\n")