#!/usr/bin/env python3
"""
HopRAG系統測試腳本
"""

import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.app.hoprag_system import HopRAGSystem, LegalNode
from backend.app.hoprag_clients import HopRAGClientManager

async def test_hoprag_system():
    """測試HopRAG系統"""
    print("🚀 開始測試HopRAG系統...")
    
    # 初始化客戶端管理器
    client_manager = HopRAGClientManager()
    
    # 獲取客戶端狀態
    status = client_manager.get_client_status()
    print(f"📊 客戶端狀態: {status}")
    
    # 初始化HopRAG系統
    hoprag_system = HopRAGSystem(
        llm_client=client_manager.get_llm_client(),
        embedding_model=client_manager.get_embedding_client()
    )
    
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
        ),
        LegalNode(
            node_id="test_item_1",
            node_type="item",
            content="本法之主管機關為經濟部。",
            contextualized_text="本法之主管機關為經濟部。",
            law_name="商標法",
            article_number="第3條",
            item_number="1",
            parent_article_id="test_article_3",
            metadata={"test": True}
        )
    ]
    
    # 添加節點到圖數據庫
    for node in test_nodes:
        hoprag_system.graph_db.add_node(node)
    
    print(f"✅ 添加了 {len(test_nodes)} 個測試節點")
    
    # 生成偽查詢
    print("🤖 生成偽查詢...")
    for node in test_nodes:
        try:
            await hoprag_system.pseudo_query_generator.generate_pseudo_queries_for_node(node)
        except Exception as e:
            print(f"❌ 節點 {node.node_id} 偽查詢生成失敗: {e}")
    
    # 構建圖邊
    print("🔗 構建圖邊...")
    try:
        await hoprag_system.graph_db.build_graph_edges()
    except Exception as e:
        print(f"❌ 圖邊構建失敗: {e}")
    
    # 獲取圖統計信息
    stats = hoprag_system.get_graph_statistics()
    print(f"📊 圖統計信息: {stats}")
    
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
    
    try:
        enhanced_results = await hoprag_system.enhanced_retrieve(
            query=test_query,
            base_results=base_results,
            k=3
        )
        
        print(f"✅ 檢索成功，獲得 {len(enhanced_results)} 個結果")
        for i, result in enumerate(enhanced_results):
            print(f"  結果 {i+1}: {result.get('node_id', 'N/A')} - {result.get('hop_level', 'N/A')}跳")
            
    except Exception as e:
        print(f"❌ 檢索失敗: {e}")
    
    print("🎉 HopRAG系統測試完成！")

if __name__ == "__main__":
    asyncio.run(test_hoprag_system())
