#!/usr/bin/env python3
"""
HopRAG模組化系統測試腳本
"""

import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.app.hoprag_system_modular import HopRAGSystem
from backend.app.hoprag_clients import HopRAGClientManager
from backend.app.hoprag_config import HopRAGConfig, DEFAULT_CONFIG, HIGH_PERFORMANCE_CONFIG
from backend.app.hoprag_graph_builder import LegalNode

async def test_modular_hoprag_system():
    """測試模組化HopRAG系統"""
    print("🚀 開始測試模組化HopRAG系統...")
    
    # 初始化客戶端管理器
    client_manager = HopRAGClientManager()
    
    # 獲取客戶端狀態
    status = client_manager.get_client_status()
    print(f"📊 客戶端狀態: {status}")
    
    # 測試不同配置
    configs_to_test = [
        ("默認配置", DEFAULT_CONFIG),
        ("高性能配置", HIGH_PERFORMANCE_CONFIG)
    ]
    
    for config_name, config in configs_to_test:
        print(f"\n🔧 測試 {config_name}...")
        
        # 初始化HopRAG系統
        hoprag_system = HopRAGSystem(
            llm_client=client_manager.get_llm_client(),
            embedding_model=client_manager.get_embedding_client(),
            config=config
        )
        
        # 測試模組狀態
        module_status = hoprag_system.get_module_status()
        print(f"📋 模組狀態: {module_status}")
        
        # 創建測試節點
        test_nodes = [
            LegalNode(
                node_id="test_article_1",
                node_type="article",
                content="第1條：為保障商標權、證明標章權、團體標章權、團體商標權及消費者利益，維護市場公平競爭，促進工商企業正常發展，特制定本法。",
                contextualized_text="第1條：為保障商標權、證明標章權、團體標章權、團體商標權及消費者利益，維護市場公平競爭，促進工商企業正常發展，特制定本法。",
                law_name="商標法",
                article_number="第1條",
                metadata={"test": True}
            ),
            LegalNode(
                node_id="test_article_2",
                node_type="article",
                content="第2條：欲取得商標權、證明標章權、團體標章權或團體商標權者，應依本法申請註冊。",
                contextualized_text="第2條：欲取得商標權、證明標章權、團體標章權或團體商標權者，應依本法申請註冊。",
                law_name="商標法",
                article_number="第2條",
                metadata={"test": True}
            )
        ]
        
        # 模擬多層次chunks數據
        multi_level_chunks = {
            "test_doc": {
                "basic_unit": [
                    {
                        "content": test_nodes[0].content,
                        "metadata": {
                            "id": test_nodes[0].node_id,
                            "law_name": test_nodes[0].law_name,
                            "article_label": test_nodes[0].article_number
                        }
                    },
                    {
                        "content": test_nodes[1].content,
                        "metadata": {
                            "id": test_nodes[1].node_id,
                            "law_name": test_nodes[1].law_name,
                            "article_label": test_nodes[1].article_number
                        }
                    }
                ]
            }
        }
        
        try:
            # 構建圖譜
            print("🏗️ 構建HopRAG圖譜...")
            await hoprag_system.build_graph_from_multi_level_chunks(multi_level_chunks)
            
            # 獲取圖統計信息
            stats = hoprag_system.get_graph_statistics()
            print(f"📊 圖統計信息: {stats}")
            
            # 測試配置更新
            print("🔧 測試配置更新...")
            new_config = HopRAGConfig(
                similarity_threshold=0.8,
                max_hops=3,
                top_k_per_hop=15
            )
            hoprag_system.update_config(new_config)
            print("✅ 配置更新成功")
            
            # 測試檢索
            print("🔍 測試檢索...")
            test_query = "商標權如何取得？"
            
            # 模擬基礎檢索結果
            base_results = [
                {
                    'node_id': 'test_article_1',
                    'content': test_nodes[0].content,
                    'similarity_score': 0.8
                }
            ]
            
            enhanced_results = await hoprag_system.enhanced_retrieve(
                query=test_query,
                base_results=base_results,
                k=3
            )
            
            print(f"✅ 檢索成功，獲得 {len(enhanced_results)} 個結果")
            for i, result in enumerate(enhanced_results):
                print(f"  結果 {i+1}: {result.get('node_id', 'N/A')} - {result.get('hop_level', 'N/A')}跳")
            
            # 測試導出/導入
            print("📤 測試導出/導入...")
            graph_data = hoprag_system.export_graph_data()
            print(f"✅ 導出成功，包含 {len(graph_data['nodes'])} 個節點")
            
            # 測試系統重置
            print("🔄 測試系統重置...")
            hoprag_system.reset_system()
            print("✅ 系統重置成功")
            
        except Exception as e:
            print(f"❌ {config_name} 測試失敗: {e}")
            continue
    
    print("\n🎉 模組化HopRAG系統測試完成！")

async def test_individual_modules():
    """測試各個模組"""
    print("\n🧪 開始測試各個模組...")
    
    from backend.app.hoprag_config import HopRAGConfig, DEFAULT_CONFIG
    from backend.app.hoprag_graph_builder import PseudoQueryGenerator, EdgeConnector, PassageGraphBuilder
    from backend.app.hoprag_hop_retriever import InitialRetriever, GraphTraverser, LLMReasoner
    from backend.app.hoprag_result_processor import RelevanceFilter, ResultRanker
    
    # 測試配置模組
    print("🔧 測試配置模組...")
    config = HopRAGConfig(similarity_threshold=0.75, max_hops=3)
    print(f"✅ 配置創建成功: {config.to_dict()}")
    
    # 測試偽查詢生成器
    print("🤖 測試偽查詢生成器...")
    from backend.app.hoprag_clients import HopRAGClientManager
    client_manager = HopRAGClientManager()
    
    pseudo_query_generator = PseudoQueryGenerator(
        llm_client=client_manager.get_llm_client(),
        config=config
    )
    
    test_node = LegalNode(
        node_id="test_node",
        node_type="article",
        content="測試內容",
        contextualized_text="測試上下文內容",
        law_name="測試法",
        article_number="第1條"
    )
    
    try:
        await pseudo_query_generator.generate_pseudo_queries_for_node(test_node)
        print(f"✅ 偽查詢生成成功: {len(test_node.incoming_questions)}個內向，{len(test_node.outgoing_questions)}個外向")
    except Exception as e:
        print(f"❌ 偽查詢生成失敗: {e}")
    
    # 測試邊連接器
    print("🔗 測試邊連接器...")
    edge_connector = EdgeConnector(config)
    print("✅ 邊連接器創建成功")
    
    # 測試結果處理器
    print("📊 測試結果處理器...")
    relevance_filter = RelevanceFilter(config)
    result_ranker = ResultRanker(config)
    print("✅ 結果處理器創建成功")
    
    print("🎉 模組測試完成！")

if __name__ == "__main__":
    asyncio.run(test_modular_hoprag_system())
    asyncio.run(test_individual_modules())
