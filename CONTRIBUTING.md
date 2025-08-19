# How to contribute

> A suggestion: if you are worried about this file, you can ask claude code to extract what you need to know and what to do before you commit your code, it is a powerful tool I think.

- [English Version](#english-version)
- [中文版](#中文版)

---

## English Version

Thank you for your interest in the DeepEcon project! This guide will help you understand how to contribute code to the project.

## Quick Start

### Environment Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/sepinetam/DeepEcon.git
   cd DeepEcon
   ```

2. **Install dependencies**
   This project uses [uv](https://docs.astral.sh/uv/) for package management:
   ```bash
   uv sync
   ```

3. **Run tests**
   ```bash
   uv run pytest
   ```

### Development Workflow

1. **Fork the project**
   - Click the "Fork" button on the GitHub page
   - Clone your fork locally

2. **Create feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Development**
   - Follow project coding standards
   - Write necessary tests
   - Ensure all tests pass

4. **Commit changes**
   ```bash
   git add .
   git commit -m "add: brief description of your changes"
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request**
   - Create PR on GitHub
   - Describe your changes and motivation
   - Wait for code review

## Coding Standards

### Python Code Style

- Use **PEP 8** standard
- All functions and variables must include type annotations
- Comments must be in English
- Use meaningful variable names

Example:
```python
from typing import List, Optional

def calculate_correlation(
    data_x: List[float], 
    data_y: List[float], 
    method: str = "pearson"
) -> Optional[float]:
    """
    Calculate correlation coefficient between two datasets.
    
    Args:
        data_x: First dataset
        data_y: Second dataset
        method: Correlation method (pearson, spearman, kendall)
    
    Returns:
        Correlation coefficient or None if calculation fails
    """
    pass
```

### Testing Requirements

- All new features must include tests
- Test files go in `tests/` directory
- Use pytest framework
- Test naming format: `test_*.py`

Run tests:
```bash
uv run pytest
uv run pytest tests/specific_test.py  # Run specific test
uv run pytest -v  # Verbose output
```

### Project Structure

```
DeepEcon/
├── src/deepecon/          # Main source code
│   ├── core/             # Core components
│   ├── estimators/       # Estimators
│   └── transforms/       # Data transforms
├── tests/                # Test files
├── releases/             # Release notes
└── docs/                 # Documentation
```

## Types of Contributions

### 1. Feature Development
- New statistical methods
- Data transformation functions
- Performance optimizations

### 2. Bug Fixes
- Fix existing issues
- Improve error handling
- Fix edge cases

### 3. Documentation Improvements
- Improve API documentation
- Add usage examples
- Improve README

### 4. Testing Enhancement
- Increase test coverage
- Add boundary tests
- Performance tests

## Commit Message Standards

Use concise and clear commit messages, format: `type: description`

Types include:
- `add`: Add new feature
- `fix`: Fix issues
- `update`: Update existing features
- `refactor`: Refactor code
- `docs`: Documentation updates
- `test`: Testing related

Examples:
```
add: add spearman correlation calculation
fix: handle edge cases in OLS estimator
docs: update API documentation for transforms
```

## Code Review

- All PRs need at least one reviewer
- Ensure CI/CD checks pass
- Respond to review comments
- Keep PR focused on single feature

## Issue Reporting

Found a bug or have feature suggestions? Please create issues in the following format:

**Bug Report**
- Problem description
- Reproduction steps
- Expected behavior
- Actual behavior
- Environment info (Python version, OS, etc.)

**Feature Request**
- Feature description
- Use case
- Expected API interface
- Reference implementation (if any)

## Development Tools

### Recommended IDE Setup
- VS Code + Python extension
- Configure Python interpreter to project virtual environment
- Enable type checking (mypy)

### Common Commands

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Format code
uv run black src/ tests/
uv run isort src/ tests/

# Type checking
uv run mypy src/

# Run specific module
uv run python -m deepecon.transforms.corr
```

## Contact

- Create GitHub Issue
- Submit Pull Request
- Email: sepinetam@gamil.com

## License

This project uses MIT license. By contributing, you agree that your contributions will be licensed under the same license.

---

## 中文版

感谢您对 DeepEcon 项目的兴趣！本指南将帮助您了解如何为项目贡献代码。

## 快速开始

### 环境准备

1. **克隆项目**
   ```bash
   git clone https://github.com/sepinetam/DeepEcon.git
   cd DeepEcon
   ```

2. **安装依赖**
   本项目使用 [uv](https://docs.astral.sh/uv/) 进行包管理：
   ```bash
   uv sync
   ```

3. **运行测试**
   ```bash
   uv run pytest
   ```

### 开发流程

1. **Fork 项目**
   - 点击 GitHub 页面右上角的 "Fork" 按钮
   - 克隆您的 fork 到本地

2. **创建功能分支**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **进行开发**
   - 遵循项目代码规范
   - 编写必要的测试
   - 确保所有测试通过

4. **提交更改**
   ```bash
   git add .
   git commit -m "add: brief description of your changes"
   git push origin feature/your-feature-name
   ```

5. **创建 Pull Request**
   - 在 GitHub 上创建 PR
   - 描述您的更改和动机
   - 等待代码审查

## 代码规范

### Python 代码风格

- 使用 **PEP 8** 标准
- 所有函数和变量必须包含类型标注
- 注释必须使用英文
- 使用有意义的变量名

示例：
```python
from typing import List, Optional

def calculate_correlation(
    data_x: List[float], 
    data_y: List[float], 
    method: str = "pearson"
) -> Optional[float]:
    """
    Calculate correlation coefficient between two datasets.
    
    Args:
        data_x: First dataset
        data_y: Second dataset
        method: Correlation method (pearson, spearman, kendall)
    
    Returns:
        Correlation coefficient or None if calculation fails
    """
    pass
```

### 测试要求

- 所有新功能必须包含测试
- 测试文件放在 `tests/` 目录下
- 使用 pytest 框架
- 测试命名格式：`test_*.py`

运行测试：
```bash
uv run pytest
uv run pytest tests/specific_test.py  # 运行特定测试
uv run pytest -v  # 详细输出
```

### 项目结构

```
DeepEcon/
├── src/deepecon/         # 主要代码
│   ├── core/             # 核心组件
│   ├── estimators/       # 计量算法
│   └── transforms/       # 数据转换
├── tests/                # 测试文件
└── source/               # 文档与示例数据
```

## 贡献类型

### 1. 功能开发
- 新的统计方法
- 数据转换功能
- 性能优化

### 2. Bug 修复
- 修复现有问题
- 改进错误处理
- 修复边界情况

### 3. 文档改进
- 完善 API 文档
- 添加使用示例
- 改进 README

### 4. 测试增强
- 提高测试覆盖率
- 添加边界测试
- 性能测试

## 提交信息规范

使用简洁明了的提交信息，格式：`type: description`

类型包括：
- `add`: 添加新功能
- `fix`: 修复问题
- `update`: 更新现有功能
- `refactor`: 重构代码
- `docs`: 文档更新
- `test`: 测试相关

示例：
```
add: add spearman correlation calculation
fix: handle edge cases in OLS estimator
docs: update API documentation for transforms
```

## 代码审查

- 所有 PR 需要至少一个审查者
- 确保 CI/CD 检查通过
- 回应审查意见
- 保持 PR 专注单一功能

## 问题报告

发现 bug 或有功能建议？请按以下格式创建 issue：

**Bug 报告**
- 问题描述
- 重现步骤
- 期望行为
- 实际行为
- 环境信息（Python版本、操作系统等）

**功能请求**
- 功能描述
- 使用场景
- 期望的 API 接口
- 参考实现（如有）

## 开发工具

### 推荐 IDE 设置
- VS Code + Python 扩展
- 配置 Python 解释器为项目虚拟环境
- 启用类型检查（mypy）

### 常用命令

```bash
# 安装依赖
uv sync

# 运行测试
uv run pytest

# 格式化代码
uv run black src/ tests/
uv run isort src/ tests/

# 类型检查
uv run mypy src/

# 运行特定模块
uv run python -m deepecon.transforms.corr
```

## 联系我们

- 创建 GitHub Issue
- 提交 Pull Request
- 发送邮件至：sepinetam@gamil.com

## 许可证

本项目采用 MIT 许可证，贡献即表示您同意您的贡献将在相同许可证下发布。

---

感谢您对 DeepEcon 的贡献！🚀