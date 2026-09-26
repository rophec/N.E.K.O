# 雀魂桌布边缘检测流程记录

本文记录当前已经验证过的桌布边缘检测流程。目标不是直接识别麻将牌，而是先把不同分辨率、不同窗口裁切下的雀魂牌桌统一变换到稳定坐标系，给后续手牌、牌河、副露识别提供可靠输入。

## 参考来源

- 参考项目：`yuhao7370/AutoMajsoul`
- 参考文件：`vendor/AutoMajsoul/Android/ingame_recognizer.py`
- 关键借鉴点：
  - HSV 颜色采样。
  - 形态学处理。
  - 从桌面区域估计四边形。
  - 使用 `cv2.getPerspectiveTransform()` 和 `cv2.warpPerspective()` 做透视变换。
  - 参考项目输出固定正方形，默认 `WARP_SIZE = 800`。

当前流程不是照抄 AutoMajsoul 的四点检测结果，而是在它的颜色筛选思想上，额外增加“最大桌布连通块 -> 外轮廓 -> 支撑边延长线 -> 交点”的步骤，以适应雀魂画面中桌布边缘被 UI、牌墙、手牌遮挡的情况。

## 核心判断

颜色筛选和形态学处理后，最大的白色连通块就是当前截图里的桌布候选区域。

后续所有边缘、延长线和透视变换都必须围绕这个最大白色连通块进行，不能从原图 Canny 边缘里直接找线。直接从原图找线会把麻将牌、牌墙、头像框、按钮等局部边缘误认为桌布边缘。

## 正常流程

每张输入图按以下步骤处理，并且每一步都单独保存图片。

### 01. 保存原图

输出：

```text
01-source.png
```

说明：

保留原始截图，作为后续所有结果的对照依据。

### 02. HSV 颜色筛选

输出：

```text
02-mask-before.png
```

步骤：

1. 将 BGR/RGB 图像转换为 HSV。
2. 按 AutoMajsoul 的思路，从画面中心偏左位置采样桌布颜色。
3. 根据采样平均 HSV 生成上下阈值。
4. 使用 `cv2.inRange()` 得到初始 mask。

当前参数：

```text
SAMPLE_FRAC = 0.10
DELTA_H = 15
DELTA_SV = 60
sample center = (image_center_x - image_width * 0.25, image_center_y)
```

### 03. 形态学处理

输出：

```text
03-mask-after.png
```

步骤：

对初始 mask 做闭运算：

```text
cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=3)
```

目的：

把桌布区域连成更稳定的大块白色区域，降低纹理、光照、UI 小缺口造成的断裂。

### 04. 取最大白色连通块

输出：

```text
04-largest-component.png
```

步骤：

1. 对 `03-mask-after.png` 调用 `cv2.connectedComponentsWithStats()`。
2. 排除背景后，选择面积最大的白色连通块。
3. 将这个连通块单独保存成二值图。

判断：

这个最大白色连通块就是桌布主体。后续不再使用其他小白块作为桌布边缘依据。

### 05. 取最大桌布区域外轮廓

输出：

```text
05-external-contour.png
```

步骤：

1. 对 `04-largest-component.png` 调用 `cv2.findContours(..., cv2.RETR_EXTERNAL, ...)`。
2. 只保留最大桌布区域的外轮廓。
3. 将外轮廓画成单独的边缘图。

重点：

这里使用的是 `RETR_EXTERNAL`，目的是避免桌布内部的牌、按钮、文字等黑洞参与边缘选线。

### 06. 从外轮廓选择四条支撑边

输出：

```text
06-selected-lines.png
```

步骤：

1. 在 `05-external-contour.png` 上运行 `cv2.HoughLinesP()`。
2. 按角度分组：
   - `top / bottom`：近似水平线，`abs(angle) <= 12`。
   - `left`：左侧负斜率线，约 `-85 <= angle <= -35`。
   - `right`：右侧正斜率线，约 `35 <= angle <= 85`。
3. 按位置和长度筛选四条支撑边。

当前选择规则：

```text
top:
  y 在图像上方 14% 内
  length >= image_width * 0.10
  选择最长的候选

bottom:
  y 在图像下方 22% 内
  length >= image_width * 0.10
  选择最长且更靠下的候选

left:
  x 在图像左侧 42% 内
  length >= image_height * 0.15
  选择最长候选

right:
  x 在图像右侧 45% 外
  length >= image_height * 0.15
  按 image_mid_y 处的 x 值选择最靠外的候选
```

说明：

右边线不能简单选择最长线。实测中最长线可能落到右侧内部缺口或牌墙边缘，应该优先选择同一高度下更靠外侧的右支撑边。

### 07. 将支撑边画回原图

输出：

```text
07-lines-on-source.png
```

说明：

把 `06-selected-lines.png` 中选出的 top、bottom、left、right 四条边画回原图，用于人工检查边是否贴合桌布外侧。

颜色约定：

```text
top    = yellow
bottom = cyan
left   = green
right  = magenta
```

### 08. 延长四条支撑边并求交点

输出：

```text
08-extended-lines.png
```

步骤：

1. 将四条线转换为标准直线方程：

```text
ax + by + c = 0
```

2. 将每条线延长到原图之外。
3. 求四个交点：

```text
top_left     = top ∩ left
top_right    = top ∩ right
bottom_left  = bottom ∩ left
bottom_right = bottom ∩ right
```

说明：

有些交点会落在原截图之外，这是正常现象。桌布边缘可能被窗口裁切，延长线交点就是对完整桌布四边形的外推。

### 09. 透视变换为固定正方形

输出：

```text
09-warp-square-800.png
```

步骤：

使用四个延长线交点作为源四边形：

```text
src = [top_left, top_right, bottom_left, bottom_right]
```

按 AutoMajsoul 的方式映射到固定正方形：

```text
dst = [[0, 0], [800, 0], [0, 800], [800, 800]]
```

调用：

```text
cv2.getPerspectiveTransform(src, dst)
cv2.warpPerspective(image, matrix, (800, 800))
```

说明：

之前试过按源四边形边长生成长方形输出，视觉上会偏扁。参考项目使用固定正方形输出，因此当前流程也采用 `800x800`，后续识别都应基于这个稳定坐标系。

### 10. 保存结构化结果

输出：

```text
summary.json
```

内容包括：

```text
source
image_size
sample_region
mean_hsv
lower_hsv
upper_hsv
largest_component
selected_lines
intersections
warp_output
```

## 已验证输出目录

### 基础验证集

输入：

```text
<source-images>\1.png
<source-images>\2.png
<source-images>\3.png
<source-images>\4.png
<source-images>\5.png
<source-images>\6.png
```

输出：

```text
<source-images>\normal_tablecloth_flow_20260626
```

结果：

`1.png` 到 `6.png` 均跑通。`牌.png` 不是同类完整桌面截图，失败原因是 top 候选不足。

### 特别验证集

输入：

```text
<source-images>\特别验证\s1.jpg
<source-images>\特别验证\s2.jpg
<source-images>\特别验证\s3.jpg
```

输出：

```text
<source-images>\特别验证\normal_tablecloth_flow
```

结果：

三张均跑通，每张都保存了 `01` 到 `09` 的逐步图片。

### 11/12/13 验证集

输入：

```text
<source-images>\11.png
<source-images>\12.png
<source-images>\13.png
```

输出：

```text
<source-images>\normal_tablecloth_flow_11_12_13
```

结果：

三张均跑通，每张都保存了 `01` 到 `09` 的逐步图片。

## 当前结论

桌布边缘检测已经基本成熟，可以作为后续 YOLO/牌识别前的标准化前处理流程。

当前稳定流程为：

```text
原图
-> HSV 颜色筛选
-> 形态学闭运算
-> 最大白色桌布连通块
-> 外轮廓
-> Hough 支撑边
-> 延长线交点
-> 800x800 透视变换
-> 后续牌识别
```

## 后续移植建议

1. 将该流程整理成插件内独立模块，例如 `perception/tablecloth_warp.py`。
2. 输入为截图帧，输出为：

```text
warped_image
source_quad
selected_lines
intersections
debug_paths
timing
```

3. 保留 debug 开关。开启时保存 `01` 到 `09` 的每一步图片。
4. 后续 YOLO 识别优先在 `09-warp-square-800.png` 坐标系中进行。
5. 对不完整截图、非牌桌截图、top/bottom 候选不足等情况返回失败原因，不要静默继续。

## 已知风险

1. 桌布颜色主题变化过大时，HSV 采样阈值可能需要自适应增强。
2. UI 或挂件如果大面积覆盖桌布，最大连通块可能被切断。
3. 当前 right 线选择依赖“右侧最外支撑线”规则，仍需要更多主题和分辨率样本验证。
4. 透视变换的底部交点可能在截图外，这是外推结果；可以用于标准化，但不能当作真实可见区域。
