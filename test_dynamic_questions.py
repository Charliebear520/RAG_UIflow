#!/usr/bin/env python3
"""
測試動態問題生成功能
"""

import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.app.hoprag_config import HopRAGConfig, DEFAULT_CONFIG
from backend.app.hoprag_graph_builder import PseudoQueryGenerator, LegalNode
from backend.app.hoprag_clients import HopRAGClientManager

async def test_dynamic_question_generation():
    """測試動態問題生成"""
    print("🚀 開始測試動態問題生成功能...")
    
    # 初始化客戶端管理器
    client_manager = HopRAGClientManager()
    
    # 創建動態配置
    dynamic_config = HopRAGConfig(
        use_dynamic_question_count=True,
        min_incoming_questions=2,
        min_outgoing_questions=4,
        max_incoming_questions=8,
        max_outgoing_questions=12
    )
    
    # 創建固定配置（對比用）
    fixed_config = HopRAGConfig(
        use_dynamic_question_count=False,
        max_pseudo_queries_per_node=5
    )
    
    # 初始化生成器
    dynamic_generator = PseudoQueryGenerator(
        llm_client=client_manager.get_llm_client(),
        config=dynamic_config
    )
    
    fixed_generator = PseudoQueryGenerator(
        llm_client=client_manager.get_llm_client(),
        config=fixed_config
    )
    
    # 創建測試節點
    test_nodes = [
        LegalNode(
            node_id="simple_article",
            node_type="article",
            content="第1條：本法所稱商標，指任何具有識別性之標識。",
            contextualized_text="第1條：本法所稱商標，指任何具有識別性之標識。",
            law_name="商標法",
            article_number="第1條",
            metadata={"test": True, "complexity": "simple"}
        ),
        LegalNode(
            node_id="complex_article",
            node_type="article",
            content="第8條：著作權人享有下列權利：一、重製權；二、公開播送權；三、公開傳輸權；四、改作權；五、編輯權；六、散布權；七、出租權；八、公開展示權；九、公開演出權；十、公開上映權。",
            contextualized_text="第8條：著作權人享有下列權利：一、重製權；二、公開播送權；三、公開傳輸權；四、改作權；五、編輯權；六、散布權；七、出租權；八、公開展示權；九、公開演出權；十、公開上映權。",
            law_name="著作權法",
            article_number="第8條",
            metadata={"test": True, "complexity": "complex"}
        )
    ]
    
    print("\n" + "="*80)
    print("📊 動態問題生成測試結果")
    print("="*80)
    
    for node in test_nodes:
        print(f"\n🔍 測試節點: {node.node_id} ({node.metadata.get('complexity', 'unknown')} complexity)")
        print(f"📝 內容: {node.content[:50]}...")
        
        # 測試動態生成
        print(f"\n🎯 動態生成模式:")
        try:
            incoming_dynamic = await dynamic_generator._generate_incoming_questions(node)
            outgoing_dynamic = await dynamic_generator._generate_outgoing_questions(node)
            
            print(f"  📥 內向問題: {len(incoming_dynamic)}個")
            for i, q in enumerate(incoming_dynamic, 1):
                print(f"    {i}. {q}")
            
            print(f"  📤 外向問題: {len(outgoing_dynamic)}個")
            for i, q in enumerate(outgoing_dynamic, 1):
                print(f"    {i}. {q}")
                
        except Exception as e:
            print(f"  ❌ 動態生成失敗: {e}")
        
        # 測試固定生成（對比）
        print(f"\n🔧 固定生成模式:")
        try:
            incoming_fixed = await fixed_generator._generate_incoming_questions(node)
            outgoing_fixed = await fixed_generator._generate_outgoing_questions(node)
            
            print(f"  📥 內向問題: {len(incoming_fixed)}個")
            for i, q in enumerate(incoming_fixed, 1):
                print(f"    {i}. {q}")
            
            print(f"  📤 外向問題: {len(outgoing_fixed)}個")
            for i, q in enumerate(outgoing_fixed, 1):
                print(f"    {i}. {q}")
                
        except Exception as e:
            print(f"  ❌ 固定生成失敗: {e}")
        
        print("\n" + "-"*60)
    
    print("\n✅ 動態問題生成測試完成！")

if __name__ == "__main__":
    asyncio.run(test_dynamic_question_generation())
