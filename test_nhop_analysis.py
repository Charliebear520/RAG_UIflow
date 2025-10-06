#!/usr/bin/env python3
"""
測試nhop分析功能
"""

import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.app.hoprag_config import HopRAGConfig, DEFAULT_CONFIG
from backend.app.hoprag_hop_retriever import Algorithm1Traverser
from backend.app.hoprag_graph_builder import LegalNode
from backend.app.hoprag_clients import HopRAGClientManager
import networkx as nx

async def test_nhop_analysis():
    """測試nhop分析功能"""
    print("🚀 開始測試nhop分析功能...")
    
    # 初始化客戶端管理器
    client_manager = HopRAGClientManager()
    
    # 創建不同nhop配置進行測試
    test_configs = [
        {
            "name": "標準配置 (nhop=4)",
            "config": HopRAGConfig(
                max_hops=4,
                enable_nhop_analysis=True,
                nhop_cost_tracking=True,
                queue_length_threshold=1.5
            )
        },
        {
            "name": "保守配置 (nhop=2)",
            "config": HopRAGConfig(
                max_hops=2,
                enable_nhop_analysis=True,
                nhop_cost_tracking=True,
                queue_length_threshold=1.5
            )
        },
        {
            "name": "激進配置 (nhop=6)",
            "config": HopRAGConfig(
                max_hops=6,
                enable_nhop_analysis=True,
                nhop_cost_tracking=True,
                queue_length_threshold=1.5
            )
        }
    ]
    
    # 創建測試圖和節點
    test_graph, test_nodes = create_test_graph()
    
    # 測試查詢
    test_queries = [
        "著作權人有哪些權利？",
        "違反商標法會有什麼後果？",
        "如何申請專利保護？"
    ]
    
    print("\n" + "="*80)
    print("📊 nhop分析測試結果")
    print("="*80)
    
    for config_info in test_configs:
        print(f"\n🔧 測試配置: {config_info['name']}")
        print("-" * 60)
        
        # 初始化遍歷器
        traverser = Algorithm1Traverser(
            llm_client=client_manager.get_llm_client(),
            embedding_model=client_manager.get_embedding_client(),
            config=config_info['config']
        )
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 測試查詢 {i}: '{query}'")
            
            try:
                # 執行檢索
                results = await traverser.reasoning_augmented_traversal(
                    query=query,
                    graph=test_graph,
                    nodes=test_nodes,
                    top_k=5,
                    n_hop=config_info['config'].max_hops
                )
                
                print(f"✅ 檢索完成，返回 {len(results)} 個結果")
                
            except Exception as e:
                print(f"❌ 檢索失敗: {e}")
        
        print("\n" + "="*60)
    
    print("\n✅ nhop分析測試完成！")

def create_test_graph():
    """創建測試圖"""
    print("🏗️ 創建測試圖...")
    
    # 創建NetworkX圖
    graph = nx.DiGraph()
    
    # 創建測試節點
    nodes = {}
    
    # 著作權法節點
    copyright_nodes = [
        LegalNode(
            node_id="copyright_art1",
            node_type="article",
            content="第8條：著作權人享有下列權利：一、重製權；二、公開播送權；三、公開傳輸權。",
            contextualized_text="第8條：著作權人享有下列權利：一、重製權；二、公開播送權；三、公開傳輸權。",
            law_name="著作權法",
            article_number="第8條",
            metadata={"test": True}
        ),
        LegalNode(
            node_id="copyright_art2",
            node_type="article",
            content="第91條：擅自以重製之方法侵害他人之著作財產權者，處三年以下有期徒刑、拘役或科或併科新臺幣七十五萬元以下罰金。",
            contextualized_text="第91條：擅自以重製之方法侵害他人之著作財產權者，處三年以下有期徒刑、拘役或科或併科新臺幣七十五萬元以下罰金。",
            law_name="著作權法",
            article_number="第91條",
            metadata={"test": True}
        )
    ]
    
    # 商標法節點
    trademark_nodes = [
        LegalNode(
            node_id="trademark_art1",
            node_type="article",
            content="第1條：本法所稱商標，指任何具有識別性之標識。",
            contextualized_text="第1條：本法所稱商標，指任何具有識別性之標識。",
            law_name="商標法",
            article_number="第1條",
            metadata={"test": True}
        ),
        LegalNode(
            node_id="trademark_art2",
            node_type="article",
            content="第95條：未得商標權人同意，為行銷目的於同一或類似之商品或服務，使用相同或近似於其註冊商標之商標，有致相關消費者混淆誤認之虞者，處三年以下有期徒刑、拘役或科或併科新臺幣二十萬元以下罰金。",
            contextualized_text="第95條：未得商標權人同意，為行銷目的於同一或類似之商品或服務，使用相同或近似於其註冊商標之商標，有致相關消費者混淆誤認之虞者，處三年以下有期徒刑、拘役或科或併科新臺幣二十萬元以下罰金。",
            law_name="商標法",
            article_number="第95條",
            metadata={"test": True}
        )
    ]
    
    # 專利法節點
    patent_nodes = [
        LegalNode(
            node_id="patent_art1",
            node_type="article",
            content="第1條：本法所稱專利，分為發明專利、新型專利及設計專利三種。",
            contextualized_text="第1條：本法所稱專利，分為發明專利、新型專利及設計專利三種。",
            law_name="專利法",
            article_number="第1條",
            metadata={"test": True}
        )
    ]
    
    # 合併所有節點
    all_nodes = copyright_nodes + trademark_nodes + patent_nodes
    
    # 添加到圖和節點字典
    for node in all_nodes:
        graph.add_node(node.node_id)
        nodes[node.node_id] = node
    
    # 添加一些測試邊
    test_edges = [
        ("copyright_art1", "copyright_art2"),
        ("trademark_art1", "trademark_art2"),
        ("copyright_art1", "trademark_art1"),
        ("trademark_art2", "patent_art1")
    ]
    
    for from_node, to_node in test_edges:
        graph.add_edge(from_node, to_node, weight=0.8)
    
    print(f"✅ 測試圖創建完成：{len(nodes)}個節點，{len(test_edges)}條邊")
    
    return graph, nodes

if __name__ == "__main__":
    asyncio.run(test_nhop_analysis())
