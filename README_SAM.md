# SAM模型分割标注使用说明

## 功能概述

视频标注工具现在集成了SAM (Segment Anything Model) 模型，可以将用户绘制的边界框作为正样本提示，对目标进行智能语义分割。

## 工作流程

### 1. 标注阶段
- 运行 `python biaozhu.py`
- 在视频第一帧上使用鼠标框选要分割的目标
- 可以框选多个目标（每个框颜色不同）
- 点击绿色"完成标注"按钮

### 2. SAM分割阶段
程序会自动：
1. 加载SAM模型（首次运行会自动下载 `sam_b.pt`）
2. 使用您绘制的边界框作为提示（bbox prompt）
3. 对每个框内的区域进行智能分割
4. 生成精确的目标掩码（mask）

### 3. 视频生成阶段
- 对视频的每一帧应用分割掩码
- 标注区域半透明高亮显示
- 输出到 `dst` 目录

## SAM模型说明

### 可用模型

1. **sam_b.pt** (默认)
   - 基础版本，推理速度快
   - 适合大多数场景
   - 自动下载大小: ~375MB

2. **sam3.pt** (可选)
   - 最新SAM3模型，更精准
   - 支持文本提示和概念分割
   - 需要从HuggingFace申请下载

### 模型配置

在 [biaozhu.py](file:///Users/jimjiang/Downloads/biaozhu/biaozhu.py) 第7行修改模型路径：

```python
SAM_MODEL_PATH = "sam_b.pt"  # 可改为 "sam3.pt" 或其他模型
```

## 技术细节

### 分割原理

SAM模型接收边界框作为视觉提示，返回：
- **掩码 (Mask)**: 二值图像，标记目标区域
- **置信度**: 分割质量评分

### 代码流程

```
用户绘制边界框
    ↓
[biaozhu.py:219-270] process_video()
    ↓
加载SAM模型
    ↓
对每个框调用: sam_model(frame, bboxes=[bbox])
    ↓
提取掩码并保存到 AnnotationBox.mask
    ↓
对每帧应用掩码渲染
    ↓
生成标注视频
```

### 关键代码

#### SAM分割调用 ([biaozhu.py:249-264](file:///Users/jimjiang/Downloads/biaozhu/biaozhu.py#L249-L264))

```python
from ultralytics import SAM

sam_model = SAM("sam_b.pt")
bbox = [box.x1, box.y1, box.x2, box.y2]
results = sam_model(frame, bboxes=[bbox], verbose=False)

if results and results[0].masks is not None:
    mask = results[0].masks.data[0].cpu().numpy()
    mask = (mask * 255).astype(np.uint8)  # 转为0-255范围
    box.mask = mask
```

#### 掩码渲染 ([biaozhu.py:78-100](file:///Users/jimjiang/Downloads/biaozhu/biaozhu.py#L78-L100))

```python
def apply_sam_mask_to_frame(self, frame, color=None):
    if color is None:
        color = self.color

    # 半透明叠加
    mask = self.mask
    colored_mask = np.zeros_like(frame)
    colored_mask[:] = color
    frame_with_mask = frame.copy()
    mask_bool = mask > 0
    frame_with_mask[mask_bool] = cv2.addWeighted(
        frame[mask_bool], 0.3,
        colored_mask[mask_bool], 0.7, 0
    )

    # 绘制轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame_with_mask, contours, -1, color, 2)

    return frame_with_mask
```

## 测试验证

运行测试脚本验证SAM集成：

```bash
python test_sam_integration.py
```

测试项目：
- ✅ SAM模型导入
- ✅ SAM模型加载
- ✅ BBox到SAM分割
- ✅ AnnotationBox掩码支持
- ✅ VideoAnnotator初始化

## 依赖说明

### 必需依赖

- `opencv-python`: 视频处理和图像显示
- `numpy`: 数值计算
- `Pillow`: 中文字体渲染
- `ultralytics`: SAM模型推理

### 安装命令

```bash
uv pip install opencv-python numpy Pillow ultralytics
```

## 注意事项

### ⚠️ 性能考虑
- SAM模型推理需要一定时间（取决于目标数量）
- 分割过程会显示进度信息
- 如果SAM模型不可用，会自动回退到简单矩形框

### ⚠️ 掩码质量
- SAM分割结果取决于边界框的准确性
- 复杂背景或重叠目标可能影响分割效果
- 可以调整边界框位置来优化结果

### ⚠️ 模型下载
- 首次运行会自动下载SAM模型
- 下载需要网络连接
- 模型文件约375MB

## 参考资料

- [Ultralytics SAM文档](https://docs.ultralytics.com/models/sam-3/)
- [SAM3 GitHub仓库](https://github.com/RizwanMunawar/sam3-inference)
- [SAM模型原理](https://arxiv.org/abs/2304.02643)
