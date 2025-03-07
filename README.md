# ComfyUI-Qwen

这是一个ComfyUI的千问API插件，支持图像和文本的多模态输入，并通过千问API进行处理。

## 功能特点

- 支持多种千问模型：
  - qwen-turbo: 通用对话模型（快速版）
  - qwen-plus: 通用对话模型（增强版）
  - qwen-max: 通用对话模型（最强版）
  - qwen-max-longcontext: 长文本对话模型
  - qwen-vl-plus: 视觉语言模型（增强版）
  - qwen-vl-max: 视觉语言模型（最强版）
  - qwen-omni-turbo: 全能模型（支持图像输入）
- 支持图像输入（可选）
  - 图像输入需要使用 VL (Vision Language) 或 Omni 模型
  - 支持 PyTorch 张量和 NumPy 数组格式的图像
- 文本输入：
  - system_prompt: 系统提示词，用于设置AI助手的角色和行为
  - prompt: 主要提示词
- 文本输出
- 使用OpenAI兼容模式调用千问API
- 可配置的参数：
  - model: 选择要使用的千问模型
  - system_prompt: 系统提示词（默认：You are a helpful assistant.）
  - prompt: 主要提示词
  - max_tokens（默认：1024）
  - temperature（默认：0.7）
  - top_p（默认：0.7）
- 灵活的依赖处理：
  - 自动检测可用的依赖
  - 与ComfyUI兼容，避免版本冲突
  - 提供清晰的错误信息，指导用户安装缺失的依赖

## 安装步骤

1. 将此仓库克隆到ComfyUI的`custom_nodes`目录下：
```bash
cd custom_nodes
git clone https://github.com/caiyi2025/comfyui-qwen.git
```

2. 安装必要的依赖：
```bash
pip install openai
```

3. 配置API密钥：
- 打开`config.json`文件
- 将`your-api-key-here`替换为您的千问API密钥

## 使用方法

1. 在ComfyUI中，您可以找到名为"Qwen AI"的节点
2. 节点输入：
   - model：选择要使用的千问模型
   - system_prompt：系统提示词，用于设置AI助手的角色
   - prompt：主要提示词
   - image：图像输入（可选，需要VL或Omni模型）
   - max_tokens（默认1024）
   - temperature（默认0.7）
   - top_p（默认0.7）
3. 节点输出：
   - 文本输出

## 模型选择指南

1. 文本对话场景：
   - 一般用途：使用 qwen-turbo
   - 更高质量：使用 qwen-plus
   - 最佳效果：使用 qwen-max
   - 长文本处理：使用 qwen-max-longcontext

2. 图像理解场景：
   - 一般用途：使用 qwen-vl-plus
   - 最佳效果：使用 qwen-vl-max
   - 全能模型：使用 qwen-omni-turbo（支持图像和文本）

## API调用说明

插件使用OpenAI兼容模式调用千问API：
- 使用OpenAI风格的API调用
- 通过兼容层访问千问API
- 支持多模态输入（文本和图像）


## 注意事项

- 使用前请确保已正确配置API密钥
- 确保选择的模型与输入类型匹配：
  - 处理图像时需要使用 VL 或 Omni 模型
- 文本输入说明：
  - system_prompt：用于设置AI助手的角色和行为方式
  - prompt：主要的提示词或问题
- 每次运行节点都会重新调用API，获取新的结果
- 插件设计为与ComfyUI兼容，避免依赖冲突 