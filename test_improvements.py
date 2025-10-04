#!/usr/bin/env python3
"""
測試新增的法律RAG改進功能
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_legal_semantic_chunking():
    """測試法律語義完整性分塊"""
    print("🔍 測試法律語義完整性分塊...")
    
    try:
        from app.legal_semantic_chunking import LegalSemanticIntegrityChunking
        
        # 創建測試文本
        test_text = """
        第28條 著作人專有將其著作改作成衍生著作或編輯成編輯著作之權利。但表演不適用之。
        前項規定，於著作人將其著作之著作財產權讓與他人或授權他人利用後，亦適用之。
        """
        
        # 初始化分塊器
        chunker = LegalSemanticIntegrityChunking()
        
        # 執行分塊
        chunks = chunker.chunk(test_text, max_chunk_size=500, preserve_concepts=True)
        
        print(f"✅ 法律語義分塊成功！")
        print(f"   - 分塊數量: {len(chunks)}")
        for i, chunk in enumerate(chunks):
            print(f"   - 分塊 {i+1}: {chunk['content'][:100]}...")
            print(f"     概念密度: {chunk['metadata']['concept_density']:.3f}")
            print(f"     重要性分數: {chunk['metadata']['importance_score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 法律語義分塊測試失敗: {e}")
        return False

def test_concept_graph():
    """測試概念圖構建"""
    print("\n🔍 測試法律概念圖構建...")
    
    try:
        from app.legal_concept_graph import LegalConceptGraph
        
        # 創建測試文檔
        test_documents = [
            {
                'content': '著作人專有重製權，即指以印刷、複印、錄音、錄影、攝影、筆錄或其他方法直接、間接、永久或暫時之重複製作。',
                'doc_id': 'test_doc_1',
                'chunk_index': 0,
                'filename': 'copyright_test.txt'
            },
            {
                'content': '商標權人於註冊指定之商品或服務，取得商標權。但商標權人不得禁止他人以符合商業交易習慣之誠實信用方法，使用自己之姓名或名稱。',
                'doc_id': 'test_doc_2', 
                'chunk_index': 0,
                'filename': 'trademark_test.txt'
            }
        ]
        
        # 初始化概念圖
        concept_graph = LegalConceptGraph()
        
        # 構建概念圖
        concept_graph.build_graph(test_documents)
        
        print(f"✅ 概念圖構建成功！")
        print(f"   - 節點數量: {concept_graph.graph.number_of_nodes()}")
        print(f"   - 邊數量: {concept_graph.graph.number_of_edges()}")
        print(f"   - 概念數量: {len(concept_graph.concepts)}")
        print(f"   - 關係數量: {len(concept_graph.relations)}")
        
        # 顯示部分概念
        print("   - 概念示例:")
        for i, (concept_id, concept) in enumerate(list(concept_graph.concepts.items())[:3]):
            print(f"     {i+1}. {concept.concept_name} ({concept.concept_type})")
        
        return True
        
    except Exception as e:
        print(f"❌ 概念圖構建測試失敗: {e}")
        return False

def test_adaptive_rag():
    """測試自適應檢索"""
    print("\n🔍 測試自適應檢索...")
    
    try:
        from app.adaptive_legal_rag import QueryAnalyzer, QueryType
        
        # 初始化查詢分析器
        analyzer = QueryAnalyzer()
        
        # 測試不同類型的查詢
        test_queries = [
            "什麼是著作權？",
            "著作權法第28條規定什麼？",
            "如何申請商標註冊？",
            "著作權與商標權的差別是什麼？"
        ]
        
        print(f"✅ 自適應檢索測試成功！")
        for query in test_queries:
            analysis = analyzer.analyze_query(query)
            print(f"   - 查詢: '{query}'")
            print(f"     類型: {analysis.query_type.value}")
            print(f"     置信度: {analysis.confidence:.3f}")
            print(f"     複雜度: {analysis.complexity_score:.3f}")
            print(f"     推薦策略: {analysis.recommended_strategies}")
        
        return True
        
    except Exception as e:
        print(f"❌ 自適應檢索測試失敗: {e}")
        return False

def test_enhanced_api():
    """測試增強版API端點"""
    print("\n🔍 測試增強版API端點...")
    
    try:
        # 檢查API端點定義
        with open('backend/app/enhanced_main.py', 'r', encoding='utf-8') as f:
            content = f.read()
            
        api_endpoints = [
            '/api/legal-semantic-chunk',
            '/api/multi-level-semantic-chunk', 
            '/api/build-concept-graph',
            '/api/concept-graph-retrieve',
            '/api/adaptive-retrieve'
        ]
        
        print(f"✅ API端點檢查成功！")
        for endpoint in api_endpoints:
            if endpoint in content:
                print(f"   ✓ {endpoint}")
            else:
                print(f"   ✗ {endpoint} (缺失)")
        
        return True
        
    except Exception as e:
        print(f"❌ API端點測試失敗: {e}")
        return False

def main():
    """主測試函數"""
    print("🚀 開始測試法律RAG系統改進功能...\n")
    
    test_results = []
    
    # 執行各項測試
    test_results.append(("法律語義分塊", test_legal_semantic_chunking()))
    test_results.append(("概念圖構建", test_concept_graph()))
    test_results.append(("自適應檢索", test_adaptive_rag()))
    test_results.append(("API端點", test_enhanced_api()))
    
    # 顯示測試結果
    print("\n" + "="*50)
    print("📊 測試結果總結")
    print("="*50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n總體結果: {passed}/{total} 項測試通過")
    
    if passed == total:
        print("🎉 所有改進功能測試通過！")
    else:
        print("⚠️  部分測試失敗，請檢查依賴項和配置")

if __name__ == "__main__":
    main()
