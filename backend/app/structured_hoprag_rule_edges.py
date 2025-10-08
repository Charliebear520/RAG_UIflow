"""
Structured-HopRAG 规则边构建器
实现4种规则边：hierarchy, reference, similar_concept, theme
"""

import re
import numpy as np
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from .structured_hoprag_config import (
    StructuredHopRAGConfig,
    EdgeType,
    LegalLevel,
    DEFAULT_CONFIG
)
from .structured_hoprag_embedding import MultiLevelNode

class RuleEdgeBuilder:
    """规则边构建器"""
    
    def __init__(self, config: StructuredHopRAGConfig = DEFAULT_CONFIG):
        self.config = config
        self.edges = []
        
    def build_all_rule_edges(
        self,
        nodes: Dict[str, MultiLevelNode]
    ) -> List[Dict[str, Any]]:
        """构建所有规则边"""
        print("🔗 开始构建规则边...")
        
        self.edges = []
        
        # 1. 层级边
        if self.config.hierarchy_edge_enabled:
            hierarchy_edges = self._build_hierarchy_edges(nodes)
            self.edges.extend(hierarchy_edges)
            print(f"  ✅ 层级边: {len(hierarchy_edges)} 条")
        
        # 2. 引用边
        if self.config.reference_edge_enabled:
            reference_edges = self._build_reference_edges(nodes)
            self.edges.extend(reference_edges)
            print(f"  ✅ 引用边: {len(reference_edges)} 条")
        
        # 3. 相似概念边
        if self.config.similar_edge_enabled:
            similar_edges = self._build_similar_concept_edges(nodes)
            self.edges.extend(similar_edges)
            print(f"  ✅ 相似概念边: {len(similar_edges)} 条")
        
        # 4. 主题边
        if self.config.theme_edge_enabled:
            theme_edges = self._build_theme_edges(nodes)
            self.edges.extend(theme_edges)
            print(f"  ✅ 主题边: {len(theme_edges)} 条")
        
        print(f"🎯 规则边构建完成，共 {len(self.edges)} 条")
        return self.edges
    
    def _build_hierarchy_edges(
        self,
        nodes: Dict[str, MultiLevelNode]
    ) -> List[Dict[str, Any]]:
        """
        构建层级边（父子关系）
        
        类型：directed
        权重：cosine_sim(parent_embed, child_embed) 或固定值1.0
        """
        edges = []
        
        for node in nodes.values():
            # 跳过没有子节点的节点
            if not node.children_ids:
                continue
            
            parent_embed = node.final_embedding
            
            for child_id in node.children_ids:
                if child_id not in nodes:
                    continue
                
                child = nodes[child_id]
                child_embed = child.final_embedding
                
                # 计算权重
                if self.config.hierarchy_weight_method == "cosine" and \
                   parent_embed is not None and child_embed is not None:
                    weight = self._cosine_similarity(parent_embed, child_embed)
                else:
                    weight = 1.0  # 固定权重
                
                edge = {
                    'from_node': node.node_id,
                    'to_node': child_id,
                    'edge_type': EdgeType.HIERARCHY.value,
                    'weight': weight,
                    'directed': True,
                    'metadata': {
                        'parent_level': node.level,
                        'child_level': child.level
                    }
                }
                edges.append(edge)
        
        return edges
    
    def _build_reference_edges(
        self,
        nodes: Dict[str, MultiLevelNode]
    ) -> List[Dict[str, Any]]:
        """
        构建引用边（准用、依第X条等）
        
        类型：directed
        权重：固定0.95（高优先级）
        """
        edges = []
        
        # 只在basic_unit（条）层级检测引用
        article_nodes = [
            n for n in nodes.values() 
            if n.level == LegalLevel.BASIC_UNIT.value
        ]
        
        for node in article_nodes:
            # 检测引用模式
            references = self._detect_references(node.content)
            
            for ref_article_num in references:
                # 查找被引用的条文节点
                target_node = self._find_article_by_number(
                    ref_article_num, 
                    nodes,
                    node.metadata.get('law_name')  # 同一法规内
                )
                
                if target_node:
                    edge = {
                        'from_node': node.node_id,
                        'to_node': target_node.node_id,
                        'edge_type': EdgeType.REFERENCE.value,
                        'weight': 0.95,
                        'directed': True,
                        'metadata': {
                            'reference_type': 'citation',
                            'cited_article': ref_article_num
                        }
                    }
                    edges.append(edge)
        
        return edges
    
    def _detect_references(self, content: str) -> List[str]:
        """检测文本中的引用"""
        references = []
        
        for pattern in self.config.reference_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                article_num = match.group(1)
                references.append(article_num)
        
        return list(set(references))  # 去重
    
    def _find_article_by_number(
        self,
        article_num: str,
        nodes: Dict[str, MultiLevelNode],
        law_name: Optional[str] = None
    ) -> Optional[MultiLevelNode]:
        """根据条文编号查找节点"""
        for node in nodes.values():
            if node.level != LegalLevel.BASIC_UNIT.value:
                continue
            
            # 检查条文编号
            node_article_num = node.metadata.get('article_number', '')
            if article_num in node_article_num or node_article_num in article_num:
                # 如果指定了法规名称，需要匹配
                if law_name:
                    node_law = node.metadata.get('law_name', '')
                    if law_name in node_law or node_law in law_name:
                        return node
                else:
                    return node
        
        return None
    
    def _build_similar_concept_edges(
        self,
        nodes: Dict[str, MultiLevelNode]
    ) -> List[Dict[str, Any]]:
        """
        构建相似概念边
        
        方法：
        1. TF-IDF提取法律关键词
        2. 词典匹配 + embedding相似度 > 0.75
        
        类型：undirected
        权重：cosine_sim
        """
        edges = []
        
        # 只在指定层级构建相似边（默认仅basic_unit）
        target_nodes = [
            n for n in nodes.values()
            if n.level in self.config.similar_edge_levels
        ]
        
        if not target_nodes:
            return edges
        
        # 提取法律术语词典
        legal_terms = self._extract_legal_terms(target_nodes)
        print(f"  📚 提取法律术语: {len(legal_terms)} 个")
        
        # 构建节点的关键词集合
        node_keywords = {}
        for node in target_nodes:
            keywords = self._extract_keywords(node.content, legal_terms)
            node_keywords[node.node_id] = keywords
        
        # 两两比较节点
        for i, node_a in enumerate(target_nodes):
            for node_b in target_nodes[i+1:]:
                # 1. 检查关键词重叠
                keywords_a = node_keywords.get(node_a.node_id, set())
                keywords_b = node_keywords.get(node_b.node_id, set())
                
                if not keywords_a or not keywords_b:
                    continue
                
                # Jaccard相似度
                jaccard_sim = len(keywords_a & keywords_b) / len(keywords_a | keywords_b)
                
                # 需要有一定的关键词重叠
                if jaccard_sim < 0.1:
                    continue
                
                # 2. 计算embedding相似度
                if node_a.final_embedding is not None and node_b.final_embedding is not None:
                    cosine_sim = self._cosine_similarity(
                        node_a.final_embedding,
                        node_b.final_embedding
                    )
                    
                    # 相似度阈值
                    if cosine_sim >= self.config.similar_edge_threshold:
                        edge = {
                            'from_node': node_a.node_id,
                            'to_node': node_b.node_id,
                            'edge_type': EdgeType.SIMILAR_CONCEPT.value,
                            'weight': cosine_sim,
                            'directed': False,
                            'metadata': {
                                'common_keywords': list(keywords_a & keywords_b),
                                'jaccard_similarity': jaccard_sim
                            }
                        }
                        edges.append(edge)
        
        return edges
    
    def _extract_legal_terms(
        self,
        nodes: List[MultiLevelNode],
        top_n: int = 100
    ) -> Set[str]:
        """使用TF-IDF提取法律术语"""
        # 收集所有文本
        documents = [node.content for node in nodes]
        
        try:
            # TF-IDF向量化
            vectorizer = TfidfVectorizer(
                max_features=top_n,
                min_df=2,  # 至少出现在2个文档中
                max_df=0.5,  # 最多出现在50%的文档中
                ngram_range=(1, 3),  # 1-3个词的短语
                token_pattern=r'[\u4e00-\u9fff]{2,}'  # 中文词汇，至少2字
            )
            
            tfidf_matrix = vectorizer.fit_transform(documents)
            feature_names = vectorizer.get_feature_names_out()
            
            # 获取高TF-IDF分数的术语
            avg_tfidf = np.mean(tfidf_matrix.toarray(), axis=0)
            top_indices = avg_tfidf.argsort()[-top_n:][::-1]
            
            legal_terms = {feature_names[i] for i in top_indices}
            
            # 添加常见法律术语
            common_legal_terms = {
                '侵权', '赔偿', '罚金', '拘役', '有期徒刑', '没收',
                '著作权', '商标权', '专利权', '合理使用', '准用',
                '民事责任', '刑事责任', '行政责任', '损害赔偿',
                '知识产权', '权利人', '义务人', '侵权人'
            }
            legal_terms.update(common_legal_terms)
            
            return legal_terms
            
        except Exception as e:
            print(f"  ⚠️ TF-IDF提取失败: {e}，使用默认术语")
            return {
                '侵权', '赔偿', '罚金', '拘役', '著作权', '商标权',
                '合理使用', '民事责任', '刑事责任'
            }
    
    def _extract_keywords(
        self,
        content: str,
        legal_terms: Set[str]
    ) -> Set[str]:
        """从内容中提取关键词（匹配法律术语）"""
        keywords = set()
        
        for term in legal_terms:
            if term in content:
                keywords.add(term)
        
        return keywords
    
    def _build_theme_edges(
        self,
        nodes: Dict[str, MultiLevelNode]
    ) -> List[Dict[str, Any]]:
        """
        构建主题边（基于聚类）
        
        方法：对高层节点（chapter/section）进行embedding聚类
        类型：undirected
        权重：聚类内相似度
        """
        edges = []
        
        # 只在指定层级构建主题边
        target_nodes = [
            n for n in nodes.values()
            if n.level in self.config.theme_levels and n.final_embedding is not None
        ]
        
        if len(target_nodes) < 2:
            return edges
        
        # 准备embedding矩阵
        embeddings = np.array([n.final_embedding for n in target_nodes])
        node_ids = [n.node_id for n in target_nodes]
        
        # K-means聚类
        try:
            n_clusters = min(self.config.theme_num_clusters, len(target_nodes))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # 按聚类分组
            clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                clusters[label].append(i)
            
            # 在每个聚类内部建边
            for cluster_id, indices in clusters.items():
                if len(indices) < 2:
                    continue
                
                # 聚类内两两连接
                for i in range(len(indices)):
                    for j in range(i+1, len(indices)):
                        idx_a = indices[i]
                        idx_b = indices[j]
                        
                        node_a_id = node_ids[idx_a]
                        node_b_id = node_ids[idx_b]
                        
                        # 计算相似度作为权重
                        similarity = self._cosine_similarity(
                            embeddings[idx_a],
                            embeddings[idx_b]
                        )
                        
                        edge = {
                            'from_node': node_a_id,
                            'to_node': node_b_id,
                            'edge_type': EdgeType.THEME.value,
                            'weight': similarity,
                            'directed': False,
                            'metadata': {
                                'cluster_id': int(cluster_id),
                                'cluster_size': len(indices)
                            }
                        }
                        edges.append(edge)
            
            print(f"  📊 主题聚类: {n_clusters} 个簇")
            
        except Exception as e:
            print(f"  ⚠️ 主题聚类失败: {e}")
        
        return edges
    
    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """计算余弦相似度"""
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(dot_product / (norm_a * norm_b))
    
    def get_edge_statistics(self) -> Dict[str, Any]:
        """获取边统计信息"""
        stats = {
            'total_edges': len(self.edges),
            'by_type': {},
            'directed_edges': sum(1 for e in self.edges if e.get('directed', False)),
            'undirected_edges': sum(1 for e in self.edges if not e.get('directed', False)),
        }
        
        # 按类型统计
        for edge in self.edges:
            edge_type = edge['edge_type']
            if edge_type not in stats['by_type']:
                stats['by_type'][edge_type] = {
                    'count': 0,
                    'avg_weight': 0.0,
                    'weights': []
                }
            stats['by_type'][edge_type]['count'] += 1
            stats['by_type'][edge_type]['weights'].append(edge['weight'])
        
        # 计算平均权重
        for edge_type in stats['by_type']:
            weights = stats['by_type'][edge_type]['weights']
            stats['by_type'][edge_type]['avg_weight'] = np.mean(weights) if weights else 0.0
            del stats['by_type'][edge_type]['weights']  # 移除原始权重列表
        
        return stats
