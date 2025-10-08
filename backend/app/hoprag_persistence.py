"""
HopRAG图谱持久化模块
自动保存和加载HopRAG图谱，避免每次重启都要重新构建
"""

import os
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

class HopRAGPersistence:
    """HopRAG图谱持久化管理器"""
    
    def __init__(self, storage_dir: str = "hoprag_storage"):
        """
        初始化持久化管理器
        
        Args:
            storage_dir: 存储目录路径
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # 文件路径
        self.graph_file = self.storage_dir / "hoprag_graph.pkl"
        self.metadata_file = self.storage_dir / "metadata.json"
        
    def save_graph(self, hoprag_system, force: bool = False) -> bool:
        """
        保存HopRAG图谱到文件
        
        Args:
            hoprag_system: HopRAG系统实例
            force: 是否强制保存（即使图谱未构建）
            
        Returns:
            是否保存成功
        """
        if not hoprag_system.is_graph_built and not force:
            print("⚠️ HopRAG图谱未构建，跳过保存")
            return False
            
        try:
            print("💾 开始保存HopRAG图谱...")
            start_time = datetime.now()
            
            # 导出图数据
            graph_data = hoprag_system.export_graph_data()
            
            # 保存到pickle文件（更快，支持复杂对象）
            with open(self.graph_file, 'wb') as f:
                pickle.dump(graph_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # 保存元数据（JSON格式，方便查看）
            metadata = {
                "save_time": start_time.isoformat(),
                "statistics": hoprag_system.get_graph_statistics(),
                "file_size_mb": os.path.getsize(self.graph_file) / (1024 * 1024)
            }
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            print(f"✅ HopRAG图谱保存成功！")
            print(f"   文件大小: {metadata['file_size_mb']:.2f} MB")
            print(f"   保存时间: {elapsed:.2f}秒")
            print(f"   位置: {self.graph_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ 保存HopRAG图谱失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_graph(self, hoprag_system) -> bool:
        """
        从文件加载HopRAG图谱
        
        Args:
            hoprag_system: HopRAG系统实例
            
        Returns:
            是否加载成功
        """
        if not self.graph_file.exists():
            print("ℹ️ 未找到已保存的HopRAG图谱")
            return False
            
        try:
            print("📂 发现已保存的HopRAG图谱，开始加载...")
            start_time = datetime.now()
            
            # 读取元数据
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    save_time = metadata.get('save_time', 'Unknown')
                    print(f"   保存时间: {save_time}")
                    print(f"   文件大小: {metadata.get('file_size_mb', 0):.2f} MB")
            
            # 从pickle文件加载
            with open(self.graph_file, 'rb') as f:
                graph_data = pickle.load(f)
            
            # 导入到HopRAG系统
            hoprag_system.import_graph_data(graph_data)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            print(f"✅ HopRAG图谱加载成功！")
            print(f"   加载时间: {elapsed:.2f}秒")
            print(f"   统计信息: {hoprag_system.get_graph_statistics()}")
            
            return True
            
        except Exception as e:
            print(f"❌ 加载HopRAG图谱失败: {e}")
            print(f"   可能需要重新构建图谱")
            import traceback
            traceback.print_exc()
            return False
    
    def has_saved_graph(self) -> bool:
        """检查是否有已保存的图谱"""
        return self.graph_file.exists()
    
    def get_metadata(self) -> Optional[Dict[str, Any]]:
        """获取已保存图谱的元数据"""
        if not self.metadata_file.exists():
            return None
            
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 读取元数据失败: {e}")
            return None
    
    def delete_saved_graph(self) -> bool:
        """删除已保存的图谱"""
        try:
            if self.graph_file.exists():
                os.remove(self.graph_file)
                print(f"✅ 已删除图谱文件: {self.graph_file}")
                
            if self.metadata_file.exists():
                os.remove(self.metadata_file)
                print(f"✅ 已删除元数据文件: {self.metadata_file}")
                
            return True
            
        except Exception as e:
            print(f"❌ 删除失败: {e}")
            return False
    
    def get_storage_info(self) -> Dict[str, Any]:
        """获取存储信息"""
        info = {
            "storage_dir": str(self.storage_dir),
            "has_saved_graph": self.has_saved_graph(),
            "graph_file": str(self.graph_file),
            "metadata_file": str(self.metadata_file)
        }
        
        if self.has_saved_graph():
            info["file_size_mb"] = os.path.getsize(self.graph_file) / (1024 * 1024)
            metadata = self.get_metadata()
            if metadata:
                info["save_time"] = metadata.get("save_time")
                info["statistics"] = metadata.get("statistics")
        
        return info
