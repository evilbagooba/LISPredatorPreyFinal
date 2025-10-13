# GitHub 使用说明简要

## 1. 上传修改后的本地代码到 GitHub

```bash
# 初始化仓库（只做一次）
git init

# 关联远程仓库
git remote add origin https://github.com/用户名/仓库名.git

# 添加 + 提交 + 推送
git add .
git commit -m "说明"
git branch -M main  # 如果是第一次推送
git push -u origin main


```


## 2. 下载远程仓库到本地
``` bash
复制
编辑
git clone https://github.com/用户名/仓库名.git

```



##  3. 回退或切换到旧版本
```bash
复制
编辑
# 查看历史提交
git log --oneline

# 临时切换到某版本
git checkout <commit-id>

# 如果需要覆盖当前版本（危险操作）
git reset --hard <commit-id>

```