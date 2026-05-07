视频标注工具 - 数据流转说明
============================

一、目录说明
------------

1. temp_data/
   - 原始标注数据目录
   - 视频切帧后的标注结果保存位置
   - 双向标注结果的追加写入位置
   - 所有 trace id 的原始记录

2. temp_data_mid/
   - 中间处理目录
   - 从 temp_data 拷贝生成，作为工作副本
   - 绿点赋值的操作目录
   - ID 映射的应用目录
   - 视频预览的渲染数据源

3. temp_data_post/
   - 最终导出目录
   - 从 temp_data_mid 导出
   - 只包含 track_id >= 999999 的标注
   - 交付给下游使用

二、数据流转
------------

┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  [1] 视频标注                                                    │
│       输入: 视频文件                                              │
│       输出: temp_data/ (frames/, labels/, annotations.json)       │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [2] 显示预览 (show_viewer)                                      │
│       temp_data/ ──拷贝──> temp_data_mid/                        │
│                           │                                      │
│                           ├── 应用 trace_id 映射                   │
│                           │                                      │
│                           └── VideoViewer 渲染 temp_data_mid      │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [3] 绿点赋值 (handle_viewer_click)                              │
│       点击 mask → 在 temp_data_mid/labels/ 中修改 track_id        │
│       同时添加到 trace_id_changes.json 映射记录                    │
│       注意: 只修改 temp_data_mid，不修改 temp_data                │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [4] ID 映射变更                                                 │
│       修改 trace_id_changes.json → 应用到 temp_data_mid/          │
│       _apply_trace_id_mappings_to_mid()                          │
│       删除映射时恢复: new_id → old_id                             │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [5] 提示帧双向标注 (do_bidirectional_inject)                     │
│       在 temp_data_mid/frames/ 上执行标注                         │
│       结果保存到 temp_data/annotations.json                       │
│       (不是 temp_data_mid，而是回写到 temp_data)                  │
│       temp_data_mid/annotations.json 同步更新                     │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [6] 删除 trace id                                               │
│       在 temp_data/ 上删除指定 track_id 的标注                    │
│       同时在 temp_data_mid/ 上恢复该映射                          │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [7] 导出 (export_to_temp_data_post)                             │
│       temp_data_mid/ ──导出──> temp_data_post/                   │
│       过滤条件: 只包含 track_id >= 999999 的标注                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

三、trace_id 颜色规则
---------------------

- track_id >= 1000000: 紫色系（浅紫 → 深紫 → 黑紫）
- track_id < 1000000: 冷色系（蓝色系）

四、关键文件
-----------

- temp_data_mid/trace_id_changes.json
  ID 映射记录文件，格式: ["ID: 100 → 1000000", "ID: 200 → 1000001", ...]

- labels/frame_XXXXXX.json
  每帧的标注数据，包含 segmentation、track_id 等字段

- annotations.json
  COCO 格式的完整标注文件
