#!/bin/bash

# 使用方式: ./print_ipynb_code.sh your_notebook.ipynb

if [ $# -ne 1 ]; then
    echo "用法: $0 <notebook.ipynb>"
    exit 1
fi

NOTEBOOK="$1"

if [ ! -f "$NOTEBOOK" ]; then
    echo "错误: 文件 '$NOTEBOOK' 不存在"
    exit 1
fi

# 提取所有 code 类型 cell 的 source 并输出
jq -r '
  .cells[] |
  select(.cell_type == "code") |
  .source[]' "$NOTEBOOK"