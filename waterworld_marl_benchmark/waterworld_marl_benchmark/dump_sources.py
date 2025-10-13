import os
import sys

# 根目录与输出文件
root = sys.argv[1] if len(sys.argv) > 1 else "."
outfile = sys.argv[2] if len(sys.argv) > 2 else "all_sources.txt"

# 要包含的扩展名
exts = {".py", ".yaml", ".yml"}
# 要排除的目录
exclude_dirs = {".git", ".venv", "venv", "__pycache__", ".ipynb_checkpoints"}

# 打开输出文件
with open(outfile, "w", encoding="utf-8") as out:
    for dirpath, dirnames, filenames in os.walk(root):
        # 过滤不希望遍历的目录
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]

        rel_dir = os.path.relpath(dirpath, root)

        for fn in sorted(filenames):
            # 获取扩展名并检查是否在包含列表中
            ext = os.path.splitext(fn)[1].lower()
            if ext not in exts:
                continue

            # 排除以 fix 开头的 Python 文件
            if ext == ".py" and fn.lower().startswith("fix"):
                continue

            path = os.path.join(dirpath, fn)
            rel = os.path.normpath(os.path.join(rel_dir, fn)) if rel_dir != '.' else fn

            out.write("=" * 80 + "\n")
            out.write(f"FILE: {rel}\n")
            out.write("=" * 80 + "\n")

            try:
                with open(path, "r", encoding="utf-8") as f:
                    out.write(f.read())
            except UnicodeDecodeError:
                # 降级为容错读取
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    out.write(f.read())
            except Exception as e:
                out.write(f"\n[⚠️ ERROR reading file: {e}]\n")

            out.write("\n\n")

print(f"✅ Done -> {outfile}")
