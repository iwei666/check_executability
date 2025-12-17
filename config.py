import os
from pathlib import Path
from threading import Lock

# 全局配置字典
config_dict: dict = {}

_config_loaded = False
_config_lock = Lock()


def load_config(config_path: str = "backend/config.yml") -> None:
    global _config_loaded, config_dict
    
    # 如果已经加载过，直接返回
    if _config_loaded:
        return
    
    # 使用锁确保线程安全
    with _config_lock:
        # 双重检查，避免多线程环境下重复加载
        if _config_loaded:
            return
        
        config_file = Path(config_path)
        if not config_file.exists():
            print(f"警告: 配置文件 {config_path} 不存在，将使用环境变量")
            _config_loaded = True
            return
        
        with open(config_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # 跳过空行和注释
                if not line or line.startswith("#"):
                    continue
                
                # 解析 KEY=VALUE 格式
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    # 存储到配置字典
                    config_dict[key] = value
                    # 如果环境变量中不存在该键，则从配置文件加载到环境变量（保持兼容）
                    if key and value and key not in os.environ:
                        os.environ[key] = value
        
        _config_loaded = True


# 模块导入时自动加载配置
load_config()

