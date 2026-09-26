# Mahjong Coach 安装失败恢复

## 症状

导入插件时只显示：

```text
plugin package cannot be installed safely
```

旧版 N.E.K.O 在 Windows 上卸载带原生依赖的插件时，`cv2.pyd` 等文件可能仍被进程占用。旧卸载流程可能先移除 `plugin.toml`，随后目录删除失败，留下没有插件身份的 `mahjong_coach` 目录。下一次导入会在插件启动之前被旧安装器的安全检查拦截，因此插件包自身无法修复这个状态。

另一种历史残留是旧的 `rename` 冲突策略产生 `mahjong_coach_1`、`mahjong_coach_2` 等重复目录。

## 当前包状态

- 当前分发包的 payload 哈希可以正常验证。
- 当前包不再携带默认 profile，不会覆盖用户的打法与捕获设置。
- 在干净目录中可以正常安装。
- 旧版宿主的安装前拦截无法通过更新 `.neko-plugin` 内部代码解决。

## 长期修复

新版 N.E.K.O 宿主包含以下恢复逻辑：

- 卸载时先把插件目录原子移动到 `.delete-pending`，再尝试清理被锁文件。
- 安装时发现缺少有效 `plugin.toml` 的同名残留目录，会先移动到 `.install-recovery`，安装失败时再回滚。
- 重新安装插件时保留已经存在的用户 profile。

这些能力必须随 N.E.K.O 应用更新分发，不能由尚未安装成功的插件注入旧宿主。

## 旧版宿主的一次性恢复

向受影响用户同时提供：

- `repair_mahjong_coach_install.cmd`
- `repair_mahjong_coach_install.ps1`
- 最新的 Mahjong Coach `.neko-plugin`

用户完全关闭 N.E.K.O 后，双击 `.cmd`。工具只处理 N.E.K.O 数据目录下名称或插件 ID 为 `mahjong_coach` 的插件目录，将其移动到：

```text
%LOCALAPPDATA%\N.E.K.O\.mahjong-coach-install-recovery\<时间戳>
```

工具不会删除文件，也不会移动 `.neko-package-profiles` 中的用户配置。恢复完成后重新启动 N.E.K.O，再导入最新插件包。

