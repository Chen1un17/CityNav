import xml.etree.ElementTree as ET
import igraph as ig
import leidenalg as la
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import json
from collections import defaultdict

def parse_sumo_net_to_igraph(net_file):
    print(f"开始解析 SUMO 文件: {net_file}...")
    start_time = time.time()
    nodes_coords = {}
    edges_info = []
    sumo_edges = {}  # 存储SUMO边信息: edge_id -> {from_node, to_node, lanes, speed}
    
    context = ET.iterparse(net_file, events=('start', 'end'))
    context = iter(context)
    event, root = next(context)
    
    for event, elem in context:
        if event == 'end':
            if elem.tag == 'junction' and elem.get('type') != 'internal':
                node_id = elem.get('id')
                x = float(elem.get('x'))
                y = float(elem.get('y'))
                nodes_coords[node_id] = {'x': x, 'y': y}
            elif elem.tag == 'edge' and 'from' in elem.attrib and 'to' in elem.attrib:
                edge_id = elem.get('id')
                from_node = elem.get('from')
                to_node = elem.get('to')
                
                # 存储SUMO边信息
                try:
                    speed = float(elem.find('lane').get('speed'))
                    num_lanes = len(elem.findall('lane'))
                    length = float(elem.find('lane').get('length'))
                    traffic_weight = speed * num_lanes
                except (AttributeError, TypeError):
                    speed = 50.0
                    num_lanes = 1
                    length = 100.0
                    traffic_weight = 1.0
                
                sumo_edges[edge_id] = {
                    'from_node': from_node,
                    'to_node': to_node,
                    'speed': speed,
                    'num_lanes': num_lanes,
                    'length': length,
                    'traffic_weight': traffic_weight
                }
                
                # 为图构建准备边信息（基于节点连接）
                edges_info.append({'source': from_node, 'target': to_node, 'W_traffic': traffic_weight})
            root.clear()
    
    print(f"文件解析完成，耗时: {time.time() - start_time:.2f} 秒。")
    print(f"共找到 {len(nodes_coords)} 个节点、{len(sumo_edges)} 条SUMO边和 {len(edges_info)} 条节点连接。")
    print("正在构建 igraph 图...")
    
    # 构建基于节点的图用于聚类
    graph = ig.Graph(directed=False)
    valid_node_ids = set()
    for edge in edges_info:
        valid_node_ids.add(edge['source'])
        valid_node_ids.add(edge['target'])
    
    nodes_coords = {node_id: coords for node_id, coords in nodes_coords.items() if node_id in valid_node_ids}
    sumo_id_to_idx = {node_id: i for i, node_id in enumerate(nodes_coords.keys())}
    
    graph.add_vertices(len(nodes_coords))
    graph.vs['id'] = list(nodes_coords.keys())
    
    edge_list = []
    traffic_weights = []
    for edge in edges_info:
        if edge['source'] in sumo_id_to_idx and edge['target'] in sumo_id_to_idx:
            source_idx = sumo_id_to_idx[edge['source']]
            target_idx = sumo_id_to_idx[edge['target']]
            edge_list.append((source_idx, target_idx))
            traffic_weights.append(edge['W_traffic'])
    
    graph.add_edges(edge_list)
    graph.es['W_traffic'] = traffic_weights
    graph.simplify(combine_edges=dict(W_traffic="mean"))
    
    print(f"igraph 图构建完成。图是连接的: {graph.is_connected()}")
    if not graph.is_connected():
        print("注意: 图不是完全连接的。算法将在最大的连通分量上运行。")
        giant_component = graph.components().giant()
        return giant_component, nodes_coords, sumo_id_to_idx, sumo_edges
    
    return graph, nodes_coords, sumo_id_to_idx, sumo_edges


# --- 2. 空间约束的 Leiden 算法 (增强日志输出) ---
def run_spatially_constrained_leiden(graph, nodes_coords, alpha=0.5, resolution=0.8, od_node_density=None, beta=0.2, smooth_od=True):
    print(f"\n{'='*20} 开始运行空间约束Leiden算法 {'='*20}")
    print(f"分析参数: alpha = {alpha}, beta(OD) = {beta}, resolution_parameter = {resolution}")
    start_time = time.time()
    
    # --- 计算空间权重 (W_spatial) ---
    spatial_weights = []
    for edge in graph.es:
        source_idx, target_idx = edge.source, edge.target
        coord1 = nodes_coords[graph.vs[source_idx]['id']]
        coord2 = nodes_coords[graph.vs[target_idx]['id']]
        distance = np.sqrt((coord1['x'] - coord2['x'])**2 + (coord1['y'] - coord2['y'])**2)
        spatial_weights.append(1.0 / (distance + 1e-6))
    graph.es['W_spatial'] = spatial_weights
    
    # --- 权重归一化与统计 ---
    W_traffic = np.array(graph.es['W_traffic'])
    W_spatial = np.array(graph.es['W_spatial'])
    
    # Min-Max归一化
    W_traffic_norm = (W_traffic - W_traffic.min()) / (W_traffic.max() - W_traffic.min() + 1e-6)
    W_spatial_norm = (W_spatial - W_spatial.min()) / (W_spatial.max() - W_spatial.min() + 1e-6)
    
    # --- 计算 OD 派生的边权重（高 OD 区域倾向于产生更多分区：降低其边权）---
    beta_effective = beta
    if od_node_density is not None:
        # 若 OD 全为 0，退回到基于交通的代理密度（节点 incident 的交通权重和，归一化）
        if all((v == 0 or v is None) for v in od_node_density.values()):
            print("OD 密度为零，启用代理密度（基于交通权重）...")
            node_proxy = defaultdict(float)
            # 准备 W_traffic 归一化供代理密度使用
            Wt = np.array(graph.es['W_traffic'])
            wmin, wmax = float(Wt.min()), float(Wt.max())
            if wmax > wmin:
                Wt_norm = (Wt - wmin) / (wmax - wmin + 1e-6)
            else:
                Wt_norm = np.ones_like(Wt)
            for e, w in zip(graph.es, Wt_norm):
                u_idx, v_idx = e.source, e.target
                u_id = graph.vs[u_idx]['id']
                v_id = graph.vs[v_idx]['id']
                node_proxy[u_id] += float(w)
                node_proxy[v_id] += float(w)
            # 归一化
            if node_proxy:
                vals = np.array(list(node_proxy.values()), dtype=float)
                vmin2, vmax2 = float(vals.min()), float(vals.max())
                if vmax2 > vmin2:
                    for k in list(node_proxy.keys()):
                        node_proxy[k] = (node_proxy[k] - vmin2) / (vmax2 - vmin2 + 1e-6)
                else:
                    for k in list(node_proxy.keys()):
                        node_proxy[k] = 0.0
            od_node_density = node_proxy
        # 邻接平滑：0.5 * 自身 + 0.5 * 邻居均值，避免全部集中到少数节点
        if smooth_od:
            neighbor_sum = defaultdict(float)
            degree_count = defaultdict(int)
            for edge in graph.es:
                u_idx, v_idx = edge.source, edge.target
                u_id = graph.vs[u_idx]['id']
                v_id = graph.vs[v_idx]['id']
                du = float(od_node_density.get(u_id, 0.0))
                dv = float(od_node_density.get(v_id, 0.0))
                neighbor_sum[u_id] += dv
                degree_count[u_id] += 1
                neighbor_sum[v_id] += du
                degree_count[v_id] += 1
            smoothed = {}
            for v in graph.vs:
                vid = v['id']
                base = float(od_node_density.get(vid, 0.0))
                if degree_count.get(vid, 0) > 0:
                    neigh_avg = neighbor_sum[vid] / float(degree_count[vid])
                else:
                    neigh_avg = 0.0
                smoothed[vid] = 0.5 * base + 0.5 * neigh_avg
            used_density = smoothed
        else:
            used_density = od_node_density

        od_edge_penalty = []
        for edge in graph.es:
            source_idx, target_idx = edge.source, edge.target
            node_u = graph.vs[source_idx]['id']
            node_v = graph.vs[target_idx]['id']
            du = used_density.get(node_u, 0.0)
            dv = used_density.get(node_v, 0.0)
            # 高OD区域希望被细分：降低该处边的聚合倾向
            # 使用负向项：高(du+dv) -> 边权更低
            pen = 1.0 - 0.5 * (du + dv)
            od_edge_penalty.append(pen)
        W_od = np.array(od_edge_penalty)
        # 归一化（鲁棒回退：若范围为0，则认为 OD 无辨识度，直接屏蔽 OD 项）
        wmin, wmax = float(W_od.min()), float(W_od.max())
        if wmax > wmin:
            W_od_norm = (W_od - wmin) / (wmax - wmin + 1e-6)
        else:
            W_od_norm = np.zeros_like(W_traffic_norm)
            beta_effective = 0.0
    else:
        W_od_norm = np.zeros_like(W_traffic_norm)
        beta_effective = 0.0

    # --- 计算复合权重 ---
    traffic_weight = max(0.0, 1.0 - alpha - beta_effective)
    # 组合：交通 + 空间 + OD负向项（抑制高OD处的边）
    W_composite = traffic_weight * W_traffic_norm + alpha * W_spatial_norm + beta_effective * W_od_norm
    
    # --- 打印权重分析 ---
    print("\n--- 权重分析 ---")
    print(f"交通权重 (原始): Min={W_traffic.min():.2f}, Max={W_traffic.max():.2f}, Mean={W_traffic.mean():.2f}")
    print(f"空间权重 (原始): Min={W_spatial.min():.4f}, Max={W_spatial.max():.2f}, Mean={W_spatial.mean():.4f}")
    if od_node_density is not None:
        print(f"OD惩罚 (归一): Min={W_od_norm.min():.2f}, Max={W_od_norm.max():.2f}, Mean={W_od_norm.mean():.2f}")
    print(f"有效权重: traffic={traffic_weight:.2f}, spatial={alpha:.2f}, od={beta_effective:.2f}")
    print(f"复合权重 (最终): Min={W_composite.min():.2f}, Max={W_composite.max():.2f}, Mean={W_composite.mean():.2f}")
    
    # --- 运行Leiden算法 ---
    print("\n--- 运行Leiden算法 ---")
    partition = la.find_partition(graph, la.RBConfigurationVertexPartition, 
                                  weights=W_composite,
                                  resolution_parameter=resolution,
                                  seed=42)
    
    # --- 打印分区结果分析 ---
    print("\n--- 结果分析 ---")
    community_sizes = partition.sizes()
    print(f"算法运行完毕，耗时: {time.time() - start_time:.2f} 秒。")
    print(f"找到的社群数量: {len(partition)}")
    print(f"分区的模块度: {partition.modularity:.4f}")
    if community_sizes:
        print(f"最大社群的规模: {max(community_sizes)} 个节点")
        print(f"最小社群的规模: {min(community_sizes)} 个节点")
        print(f"社群规模分布 (前10): {sorted(community_sizes, reverse=True)[:10]}")
    print(f"{'='*25} 分析结束 {'='*25}\n")
    return partition

def run_leiden_with_escalation(graph, nodes_coords, alpha, beta, od_node_density,
                               base_resolution=1.0, min_clusters=100, max_tries=6, factor=1.5):
    """
    自适应提升分辨率，直到初始簇数达到期望下限（过分割初始化），以便后续仅用“合并”得到均衡规模。
    """
    resolution = float(base_resolution)
    best = None
    for t in range(1, max_tries + 1):
        print(f"[初始化] 尝试 {t}/{max_tries}，resolution={resolution:.4f}")
        part = run_spatially_constrained_leiden(
            graph, nodes_coords, alpha=alpha, resolution=resolution,
            od_node_density=od_node_density, beta=beta
        )
        num = len(part)
        print(f"[初始化] 当前簇数={num}（期望≥{min_clusters}）")
        best = part
        if num >= min_clusters:
            break
        resolution *= factor
    return best


# --- 3. 可视化 ---
def visualize_partition(graph, partition, nodes_coords, alpha):
    print("\n开始生成可视化结果...")
    
    # 创建布局 (直接使用节点坐标)
    layout_coords = [[nodes_coords[v['id']]['x'], -nodes_coords[v['id']]['y']] for v in graph.vs]
    layout = ig.Layout(coords=layout_coords)

    # 支持直接传入 membership 列表或 VertexClustering
    if hasattr(partition, 'membership'):
        membership = partition.membership
    else:
        membership = partition
    unique_labels = sorted(set(membership))
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    mapped_membership = [label_to_idx[m] for m in membership]
    num_communities = len(unique_labels)
    palette = plt.get_cmap('gist_rainbow')
    community_colors = [palette(i / max(1, num_communities)) for i in range(num_communities)]
    random.shuffle(community_colors)
    vertex_colors = [community_colors[m] for m in mapped_membership]
    
    fig, ax = plt.subplots(figsize=(20, 20))
    
    ig.plot(
        graph,
        target=ax,
        layout=layout,
        vertex_size=10,            
        vertex_color=vertex_colors,
        vertex_frame_width=0,      
        vertex_label=None,
        edge_width=0.5,
        edge_color='#CCCCCC'       #
    )
    
    ax.set_title(f'NYC Road Network Clustering (alpha={alpha})\nFound {num_communities} communities', fontsize=20)
    ax.set_xticks([])
    ax.set_yticks([])
    
    output_filename = f'nyc_cluster_alpha_{alpha}.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    print(f"可视化结果已保存为: {output_filename}")
    plt.show()

def parse_sumo_routes_od(rou_file, sumo_edges):
    """
    解析 SUMO 路由/OD 文件，统计以节点为单位的出发/到达次数
    支持 <trip>, <flow>, <vehicle><route edges="..."> 三种常见形式
    """
    print(f"开始解析 OD 路由文件: {rou_file}...")
    start_time = time.time()

    node_origin_counts = defaultdict(float)
    node_destination_counts = defaultdict(float)
    # TAZ -> 边集合（taz 内的出入口边）
    taz_edges_map = defaultdict(set)
    # 路由定义与分布
    route_id_to_edges = {}
    route_dist_map = {}
    # 统计
    cnt_trip = cnt_flow = cnt_vehicle = 0
    cnt_rd_used = cnt_route_ref = cnt_route_inline = 0

    # 建立 edge -> (from_node, to_node) 映射
    edge_to_endpoints = {eid: (einfo['from_node'], einfo['to_node']) for eid, einfo in sumo_edges.items()}

    def edge_to_origin_node(edge_id):
        if edge_id in edge_to_endpoints:
            return edge_to_endpoints[edge_id][0]
        return None

    def edge_to_destination_node(edge_id):
        if edge_id in edge_to_endpoints:
            return edge_to_endpoints[edge_id][1]
        return None

    context = ET.iterparse(rou_file, events=("start", "end"))
    context = iter(context)
    try:
        event, root = next(context)
    except StopIteration:
        print("OD 文件为空或格式异常。")
        return {}, {}, {}

    def add_od_by_edges(from_edge, to_edge, weight=1.0):
        if not from_edge or not to_edge:
            return
        on = edge_to_origin_node(from_edge)
        dn = edge_to_destination_node(to_edge)
        if on is not None:
            node_origin_counts[on] += float(weight)
        if dn is not None:
            node_destination_counts[dn] += float(weight)

    def add_od_by_taz(from_taz, to_taz, weight=1.0):
        if not from_taz or not to_taz:
            return
        from_edges = list(taz_edges_map.get(from_taz, []))
        to_edges = list(taz_edges_map.get(to_taz, []))
        if not from_edges and not to_edges:
            return
        # 将权重平均分配到该 TAZ 的所有出入口边
        if from_edges:
            per_from = float(weight) / float(len(from_edges))
            for fe in from_edges:
                on = edge_to_origin_node(fe)
                if on is not None:
                    node_origin_counts[on] += per_from
        if to_edges:
            per_to = float(weight) / float(len(to_edges))
            for te in to_edges:
                dn = edge_to_destination_node(te)
                if dn is not None:
                    node_destination_counts[dn] += per_to

    def add_od_by_route_edges_seq(edges_seq, weight=1.0):
        if not edges_seq:
            return
        # 过滤内部边（以 ':' 开头）并确保在 sumo_edges 中
        filtered = [e for e in edges_seq if not e.startswith(':') and e in edge_to_endpoints]
        if not filtered:
            return
        from_edge = filtered[0]
        to_edge = filtered[-1]
        add_od_by_edges(from_edge, to_edge, weight)

    def parse_route_distribution_element(rd_elem):
        entries = []  # list of (edges_seq, weight)
        # 首先收集所有 route 子元素
        children = list(rd_elem)
        probs = []
        for ch in children:
            if ch.tag != 'route':
                continue
            prob = ch.get('probability') or ch.get('prob') or ch.get('weight') or ch.get('p')
            try:
                prob = float(prob) if prob is not None else None
            except Exception:
                prob = None
            edges_attr = ch.get('edges')
            if edges_attr:
                edges_seq = edges_attr.strip().split()
            else:
                # 可能通过 id/refId 引用已有 route
                ref_id = ch.get('refId') or ch.get('id') or ch.get('route')
                edges_seq = route_id_to_edges.get(ref_id, [])
            if edges_seq:
                entries.append((edges_seq, prob))
                probs.append(prob)
        # 归一化概率（若缺失则均分）
        if entries:
            if any(p is None for _, p in entries):
                n = len(entries)
                entries = [(seq, 1.0 / n) for seq, _ in entries]
            else:
                s = sum(p for _, p in entries)
                if s > 0:
                    entries = [(seq, p / s) for seq, p in entries]
                else:
                    n = len(entries)
                    entries = [(seq, 1.0 / n) for seq, _ in entries]
        return entries

    def apply_route_ref_or_dist(elem, base_weight):
        nonlocal cnt_rd_used, cnt_route_ref, cnt_route_inline
        # 1) 直接引用 route id
        rid = elem.get('route')
        if rid and rid in route_id_to_edges:
            add_od_by_route_edges_seq(route_id_to_edges[rid], base_weight)
            cnt_route_ref += 1
            return True
        # 2) 引用 routeDistribution id
        rdid = elem.get('routeDistribution') or elem.get('routeDist')
        if rdid and rdid in route_dist_map:
            for edges_seq, prob in route_dist_map[rdid]:
                add_od_by_route_edges_seq(edges_seq, base_weight * float(prob))
            cnt_rd_used += 1
            return True
        # 3) 内嵌 route 元素（单条）
        route_child = elem.find('route')
        if route_child is not None and (route_child.get('edges') or route_child.get('id')):
            edges_seq = []
            if route_child.get('edges'):
                edges_seq = route_child.get('edges').strip().split()
            else:
                ref = route_child.get('id')
                edges_seq = route_id_to_edges.get(ref, [])
            if edges_seq:
                add_od_by_route_edges_seq(edges_seq, base_weight)
                cnt_route_inline += 1
                return True
        # 4) 内嵌 routeDistribution 元素
        rd_child = elem.find('routeDistribution')
        if rd_child is not None:
            entries = parse_route_distribution_element(rd_child)
            for edges_seq, prob in entries:
                add_od_by_route_edges_seq(edges_seq, base_weight * float(prob))
            cnt_rd_used += 1
            return True
        return False

    for event, elem in context:
        if event == 'end':
            tag = elem.tag
            try:
                if tag == 'route':
                    rid = elem.get('id')
                    edges_attr = elem.get('edges')
                    if rid and edges_attr:
                        route_id_to_edges[rid] = edges_attr.strip().split()
                elif tag == 'routeDistribution':
                    rdid = elem.get('id')
                    if rdid:
                        route_dist_map[rdid] = parse_route_distribution_element(elem)
                elif tag == 'taz':
                    taz_id = elem.get('id')
                    if taz_id:
                        edges_attr = elem.get('edges') or elem.get('sources') or elem.get('sinks')
                        if edges_attr:
                            for eid in edges_attr.strip().split():
                                taz_edges_map[taz_id].add(eid)
                        # 解析子元素 tazSource / tazSink
                        for ch in list(elem):
                            if ch.tag in ('tazSource', 'tazSink', 'source', 'sink'):
                                e = ch.get('edges') or ch.get('id')
                                if e:
                                    for eid in e.strip().split():
                                        taz_edges_map[taz_id].add(eid)
                elif tag == 'trip':
                    from_edge = elem.get('from') or elem.get('fromEdge')
                    to_edge = elem.get('to') or elem.get('toEdge')
                    from_taz = elem.get('fromTaz')
                    to_taz = elem.get('toTaz')
                    used = apply_route_ref_or_dist(elem, base_weight=1.0)
                    if not used:
                        if from_edge or to_edge:
                            add_od_by_edges(from_edge, to_edge, 1.0)
                        elif from_taz or to_taz:
                            add_od_by_taz(from_taz, to_taz, 1.0)
                    cnt_trip += 1
                elif tag == 'flow':
                    from_edge = elem.get('from') or elem.get('fromEdge')
                    to_edge = elem.get('to') or elem.get('toEdge')
                    from_taz = elem.get('fromTaz')
                    to_taz = elem.get('toTaz')
                    # flow 的权重估计：优先使用 number；否则 vehsPerHour * 持续时间
                    weight = 1.0
                    number = elem.get('number')
                    if number is not None:
                        try:
                            weight = float(number)
                        except ValueError:
                            weight = 1.0
                    else:
                        vph = elem.get('vehsPerHour')
                        begin = elem.get('begin')
                        end = elem.get('end')
                        prob = elem.get('probability') or elem.get('prob')
                        try:
                            vph = float(vph) if vph is not None else None
                            begin = float(begin) if begin is not None else None
                            end = float(end) if end is not None else None
                            if prob is not None:
                                prob = float(prob)
                                if begin is not None and end is not None and end > begin:
                                    weight = prob * (end - begin)
                                else:
                                    weight = prob
                            elif vph is not None and begin is not None and end is not None and end > begin:
                                weight = vph * (end - begin) / 3600.0
                        except Exception:
                            weight = 1.0
                    used = apply_route_ref_or_dist(elem, base_weight=weight)
                    if not used:
                        if from_edge or to_edge:
                            add_od_by_edges(from_edge, to_edge, weight)
                        elif from_taz or to_taz:
                            add_od_by_taz(from_taz, to_taz, weight)
                    cnt_flow += 1
                elif tag == 'vehicle':
                    # vehicle 内嵌的 route
                    used = apply_route_ref_or_dist(elem, base_weight=1.0)
                    if not used:
                        route_elem = elem.find('route')
                        if route_elem is not None and (route_elem.get('edges') or route_elem.get('id')):
                            if route_elem.get('edges'):
                                edges_seq = route_elem.get('edges').strip().split()
                            else:
                                edges_seq = route_id_to_edges.get(route_elem.get('id'), [])
                            add_od_by_route_edges_seq(edges_seq, 1.0)
                    cnt_vehicle += 1
            finally:
                root.clear()

    # 计算节点 OD 密度并归一化
    node_total_od = {}
    for nid in set(list(node_origin_counts.keys()) + list(node_destination_counts.keys())):
        node_total_od[nid] = node_origin_counts.get(nid, 0.0) + node_destination_counts.get(nid, 0.0)

    if node_total_od:
        values = np.array(list(node_total_od.values()), dtype=float)
        vmin, vmax = float(values.min()), float(values.max())
        if vmax > vmin:
            for nid in node_total_od:
                node_total_od[nid] = (node_total_od[nid] - vmin) / (vmax - vmin + 1e-6)
        else:
            for nid in node_total_od:
                node_total_od[nid] = 0.0

    # 日志与样本
    nonzero_O = sum(1 for v in node_origin_counts.values() if v > 0)
    nonzero_D = sum(1 for v in node_destination_counts.values() if v > 0)
    print(f"OD 解析完成，存在起讫点的节点数: {len(node_total_od)}，TAZ数: {len(taz_edges_map)}，耗时: {time.time() - start_time:.2f} 秒")
    print(f"统计: trips={cnt_trip}, flows={cnt_flow}, vehicles={cnt_vehicle}, routeRef使用={cnt_route_ref}, routeDistribution使用={cnt_rd_used}, inlineRoute={cnt_route_inline}")
    print(f"非零起点节点数={nonzero_O}, 非零终点节点数={nonzero_D}")
    # 显示前5个 O/D 最大的节点
    def top_items(d, k=5):
        return sorted(d.items(), key=lambda kv: -kv[1])[:k]
    print(f"Top-5 O 节点: {top_items(node_origin_counts)[:5]}")
    print(f"Top-5 D 节点: {top_items(node_destination_counts)[:5]}")
    return dict(node_origin_counts), dict(node_destination_counts), node_total_od

def iterative_merge_regions(graph, initial_membership, nodes_coords, node_origin_counts, node_destination_counts,
                            max_regions=50, min_regions=10, desired_regions=50,
                            imbalance_threshold=0.15, size_tolerance=0.25, max_iterations=200,
                            log_progress=True):
    """
    基于 OD 的迭代合并：
    - 优先合并 OD 总量较低的区域
    - 对于不平衡的区域（|O-D|/(O+D) > 阈值），与相邻区域合并以降低不平衡
    直到区域数 <= max_regions 且各区域不平衡度 <= 阈值，或达到迭代上限
    返回合并后的 membership（标签连续化）
    """

    # 复制 membership
    if hasattr(initial_membership, '__iter__'):
        membership = list(initial_membership)
    else:
        membership = list(initial_membership.membership)

    def relabel_membership(mb):
        labels = sorted(set(mb))
        mapping = {old: i for i, old in enumerate(labels)}
        return [mapping[x] for x in mb]

    def compute_region_stats_and_adjacency(mb):
        regions = sorted(set(mb))
        region_nodes = {r: [] for r in regions}
        for idx, label in enumerate(mb):
            region_nodes[label].append(idx)
        region_stats = {}
        for r, node_indices in region_nodes.items():
            xs, ys = [], []
            O_sum, D_sum = 0.0, 0.0
            for vi in node_indices:
                node_id = graph.vs[vi]['id']
                coord = nodes_coords[node_id]
                xs.append(coord['x'])
                ys.append(coord['y'])
                O_sum += float(node_origin_counts.get(node_id, 0.0))
                D_sum += float(node_destination_counts.get(node_id, 0.0))
            cx = float(np.mean(xs)) if xs else 0.0
            cy = float(np.mean(ys)) if ys else 0.0
            total = O_sum + D_sum
            imbalance = abs(O_sum - D_sum) / (total + 1e-6)
            region_stats[r] = {
                'nodes': node_indices,
                'centroid': (cx, cy),
                'O': O_sum,
                'D': D_sum,
                'total': total,
                'imbalance': imbalance,
                'node_count': len(node_indices),
                'edge_count_internal': 0,
                'edge_count_boundary': 0,
            }

        adjacency = defaultdict(set)
        total_internal_edges = 0
        for e in graph.es:
            u, v = e.source, e.target
            ru, rv = mb[u], mb[v]
            if ru == rv:
                region_stats[ru]['edge_count_internal'] += 1
                total_internal_edges += 1
            else:
                region_stats[ru]['edge_count_boundary'] += 1
                region_stats[rv]['edge_count_boundary'] += 1
                adjacency[ru].add(rv)
                adjacency[rv].add(ru)

        return region_stats, adjacency, total_internal_edges

    def all_balanced(stats):
        for r, s in stats.items():
            if s['imbalance'] > imbalance_threshold:
                return False
        return True

    membership = relabel_membership(membership)
    # 全局节点与边目标（基于内部边）
    total_nodes = len(membership)
    iter_count = 0
    while iter_count < max_iterations:
        iter_count += 1
        stats, adjacency, total_internal_edges = compute_region_stats_and_adjacency(membership)
        regions = sorted(stats.keys())
        # 以“目标区域数”推导边规模目标
        current_region_target = min(max(min_regions, len(regions)), max_regions)
        target_edges = max(1.0, total_internal_edges / float(desired_regions))
        lower_bound = (1.0 - size_tolerance) * target_edges
        upper_bound = (1.0 + size_tolerance) * target_edges
        # 记录进度
        if log_progress:
            sizes = [stats[r]['node_count'] for r in regions]
            esizes = [stats[r]['edge_count_internal'] for r in regions]
            imbs = [stats[r]['imbalance'] for r in regions]
            print(f"[迭代 {iter_count}] 区域数={len(regions)} | 目标内部边≈{target_edges:.1f} | 边规模 min/mean/max = {min(esizes)}/{np.mean(esizes):.1f}/{max(esizes)} | 节点规模 min/mean/max = {min(sizes)}/{np.mean(sizes):.1f}/{max(sizes)} | 不平衡 max = {max(imbs):.3f}")

        # 满足条件停止：区域数不超过 max_regions，且规模在容差范围内，且不平衡达标
        sizes_ok = (min([stats[r]['edge_count_internal'] for r in regions]) >= lower_bound and max([stats[r]['edge_count_internal'] for r in regions]) <= upper_bound)
        if len(regions) <= max_regions and len(regions) <= desired_regions and all_balanced(stats) and sizes_ok:
            print(f"迭代合并完成：区域数={len(regions)}，满足不平衡阈值={imbalance_threshold} 且内部边规模均衡")
            break
        if len(regions) <= min_regions:
            print(f"已达到最小区域数 {min_regions}，停止进一步合并。")
            break

        # 选择合并候选：严格“小与小合并、低OD与低OD合并”
        # 排序键：是否偏小 -> 区域OD总量升序 -> 不平衡度降序
        def is_small(r):
            return stats[r]['edge_count_internal'] < lower_bound
        candidates = sorted(regions, key=lambda r: (not is_small(r), stats[r]['total'], -stats[r]['imbalance']))

        used = set()
        merge_pairs = []
        # 限制本轮最多合并对数，避免越过目标区域数
        max_pairs_this_round = max(0, len(regions) - desired_regions)
        for r in candidates:
            if r in used:
                continue
            neighbors = [n for n in adjacency.get(r, []) if n not in used and n != r]
            if not neighbors:
                continue
            # 选择使规模更接近目标、且不平衡改善更大的邻居
            best_neighbor = None
            best_score = -1e18
            rx, ry = stats[r]['centroid']
            for s in neighbors:
                # 仅允许“小与小合并”：两个区域都在下界之下
                if not (stats[r]['edge_count_internal'] < target_edges and stats[s]['edge_count_internal'] < target_edges):
                    continue
                merged_edges = stats[r]['edge_count_internal'] + stats[s]['edge_count_internal']
                # 合并后仍不应超过上界
                if merged_edges > upper_bound:
                    continue
                sx, sy = stats[s]['centroid']
                dist = np.hypot(rx - sx, ry - sy) + 1e-6
                O_new = stats[r]['O'] + stats[s]['O']
                D_new = stats[r]['D'] + stats[s]['D']
                total_new = O_new + D_new
                imb_new = abs(O_new - D_new) / (total_new + 1e-6)
                improve = (max(stats[r]['imbalance'], stats[s]['imbalance']) - imb_new)
                # 低OD优先合并：区域总OD越小越好
                od_penalty = 0.0005 * (stats[r]['total'] + stats[s]['total'])
                size_closeness = -abs(merged_edges - target_edges) / (target_edges + 1e-6)
                # 评分：规模接近目标（主导） + 不平衡改善 - 距离 - OD 惩罚
                score = 2.0 * size_closeness + 1.0 * improve - 0.01 * dist - od_penalty
                if score > best_score:
                    best_score = score
                    best_neighbor = s
            if best_neighbor is not None:
                merge_pairs.append((r, best_neighbor))
                used.add(r)
                used.add(best_neighbor)
                if len(merge_pairs) >= max_pairs_this_round:
                    break

        # 不再放宽上界；若没有合并对，直接结束以避免产生大区
        if not merge_pairs:
            print("没有符合“小与小、低OD与低OD”的合并候选，结束迭代。")
            break
        if len(merge_pairs) >= max_pairs_this_round:
            break

        if not merge_pairs:
            print("没有可改进的合并候选，提前结束迭代。")
            break

        # 应用所有合并（并行，不互相冲突）
        if log_progress:
            print(f"[迭代 {iter_count}] 将执行合并对数: {len(merge_pairs)}")
        for (target_label, source_label) in merge_pairs:
            for i in range(len(membership)):
                if membership[i] == source_label:
                    membership[i] = target_label
        membership = relabel_membership(membership)
        if log_progress:
            new_region_count = len(set(membership))
            print(f"[迭代 {iter_count}] 合并完成后区域数: {new_region_count}")

    return membership

def merge_regions_by_low_od(graph, membership, nodes_coords, node_origin_counts, node_destination_counts,
                            desired_regions=50, max_iterations=100, log_progress=True):
    """
    仅依据区域 O+D 总量进行合并：
    - 每次选择 O+D 最小的两个相邻区域合并（若无相邻，则选距离最近的两个区域）
    - 直到区域数 <= desired_regions 或达到迭代上限
    """
    def relabel_membership(mb):
        labels = sorted(set(mb))
        mapping = {old: i for i, old in enumerate(labels)}
        return [mapping[x] for x in mb]

    def compute_stats_and_adjacency(mb):
        regions = sorted(set(mb))
        region_nodes = {r: [] for r in regions}
        for idx, label in enumerate(mb):
            region_nodes[label].append(idx)
        stats = {}
        for r, node_indices in region_nodes.items():
            xs, ys = [], []
            O_sum, D_sum = 0.0, 0.0
            for vi in node_indices:
                nid = graph.vs[vi]['id']
                coord = nodes_coords[nid]
                xs.append(coord['x'])
                ys.append(coord['y'])
                O_sum += float(node_origin_counts.get(nid, 0.0))
                D_sum += float(node_destination_counts.get(nid, 0.0))
            cx = float(np.mean(xs)) if xs else 0.0
            cy = float(np.mean(ys)) if ys else 0.0
            stats[r] = {
                'nodes': node_indices,
                'centroid': (cx, cy),
                'od_total': O_sum + D_sum,
                'O': O_sum,
                'D': D_sum,
            }
        adjacency = defaultdict(set)
        for e in graph.es:
            u, v = e.source, e.target
            ru, rv = mb[u], mb[v]
            if ru != rv:
                adjacency[ru].add(rv)
                adjacency[rv].add(ru)
        return stats, adjacency

    membership = list(membership)
    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        stats, adjacency = compute_stats_and_adjacency(membership)
        regions = sorted(stats.keys())
        if len(regions) <= desired_regions:
            if log_progress:
                print(f"[OD合并] 已达到目标区域数: {len(regions)}")
            break

        # 候选相邻对：按合并后 od_total 升序（偏向小+小）
        pairs = []
        for r in regions:
            for s in adjacency.get(r, []):
                if r < s:
                    pairs.append((stats[r]['od_total'] + stats[s]['od_total'], r, s))
        if pairs:
            pairs.sort(key=lambda x: x[0])
            _, r_sel, s_sel = pairs[0]
        else:
            # 没有相邻对：按质心距离选最近对
            best = None
            for i, r in enumerate(regions):
                rx, ry = stats[r]['centroid']
                for s in regions[i+1:]:
                    sx, sy = stats[s]['centroid']
                    dist = np.hypot(rx - sx, ry - sy)
                    score = (stats[r]['od_total'] + stats[s]['od_total'], dist)
                    if best is None or score < best[0]:
                        best = (score, r, s)
            _, r_sel, s_sel = best

        if log_progress:
            print(f"[OD合并] 合并区域 {r_sel} + {s_sel}，合并前各自OD: {stats[r_sel]['od_total']:.1f}, {stats[s_sel]['od_total']:.1f}")
        # 执行合并：将 s_sel 并入 r_sel
        for i in range(len(membership)):
            if membership[i] == s_sel:
                membership[i] = r_sel
        membership = relabel_membership(membership)

    return membership

def balance_regions_by_od(graph, membership, node_origin_counts, node_destination_counts,
                          od_balance_tolerance=0.10, max_passes=20, max_moves_per_pass=20000,
                          consider_interior_ring=True, interior_ring_depth=1,
                          max_hops_for_target=2, allow_global_target=True,
                          preserve_connectivity=True,
                          log_progress=True):
    """
    在固定区域数量的前提下，通过将边界节点从 O/D 过高的区域移动到 O/D 过低的相邻区域，实现每区 O 与 D 总量均衡。
    - 目标：使每个区域的 O 与 D 分别接近全局均值，容忍度由 od_balance_tolerance 决定
    - 策略：贪心地移动能最大幅度降低 (O偏差^2 + D偏差^2) 的边界节点
    - 不保证拓扑连通性，但优先移动边界节点以减少断裂概率
    - 可选：将候选节点扩展为边界向内的多层“内圈”（interior_ring_depth 层），以搬运更深处的高 OD 节点
    """

    # 准备邻接表
    num_vertices = graph.vcount()
    neighbors = [[] for _ in range(num_vertices)]
    for e in graph.es:
        u, v = e.source, e.target
        neighbors[u].append(v)
        neighbors[v].append(u)

    regions = sorted(set(membership))
    region_count = len(regions)
    region_index = {r: i for i, r in enumerate(regions)}

    # 节点 O/D
    node_O = np.zeros(num_vertices, dtype=float)
    node_D = np.zeros(num_vertices, dtype=float)
    for i in range(num_vertices):
        nid = graph.vs[i]['id']
        node_O[i] = float(node_origin_counts.get(nid, 0.0))
        node_D[i] = float(node_destination_counts.get(nid, 0.0))

    # 区域 O/D 汇总
    def compute_region_od(mb):
        O = np.zeros(region_count, dtype=float)
        D = np.zeros(region_count, dtype=float)
        for i in range(num_vertices):
            rid = region_index[mb[i]]
            O[rid] += node_O[i]
            D[rid] += node_D[i]
        return O, D

    def build_boundary_nodes(mb):
        # 返回：每个区域的边界节点列表，以及每个节点的1跳可达相邻区域集合
        region_to_boundary = {r: [] for r in regions}
        node_to_neighbor_regions_1hop = [set() for _ in range(num_vertices)]
        for i in range(num_vertices):
            r = mb[i]
            for j in neighbors[i]:
                r2 = mb[j]
                if r2 != r:
                    region_to_boundary[r].append(i)
                    node_to_neighbor_regions_1hop[i].add(r2)
        return region_to_boundary, node_to_neighbor_regions_1hop

    def sse(O, D, Ot, Dt):
        return float(np.sum((O - Ot) ** 2 + (D - Dt) ** 2))

    O, D = compute_region_od(membership)
    Ot = np.full(region_count, np.sum(O) / max(1, region_count))
    Dt = np.full(region_count, np.sum(D) / max(1, region_count))
    tol_O = od_balance_tolerance * (Ot[0] if Ot.size > 0 else 1.0)
    tol_D = od_balance_tolerance * (Dt[0] if Dt.size > 0 else 1.0)
    current_sse = sse(O, D, Ot, Dt)

    def within_tolerance(O, D):
        if region_count == 0:
            return True
        return (np.max(np.abs(O - Ot)) <= tol_O) and (np.max(np.abs(D - Dt)) <= tol_D)

    passes = 0
    while passes < max_passes and not within_tolerance(O, D):
        passes += 1
        moves = 0
        region_to_boundary, node_to_neighbor_regions_1hop = build_boundary_nodes(membership)

        # 按 O+D 超标程度从大到小排序区域
        excess = (O - Ot) + (D - Dt)
        order = np.argsort(-excess)
        # 预先选取若干 O/D 明显不足的候选区域（全局）
        deficit = (Ot - O) + (Dt - D)
        deficit_idxs = list(np.argsort(-deficit))  # 从缺口大的到小的
        top_deficit_regions = [regions[idx] for idx in deficit_idxs[:8]]
        for ridx in order:
            r = regions[ridx]
            if O[ridx] <= Ot[ridx] + tol_O and D[ridx] <= Dt[ridx] + tol_D:
                continue
            # 候选节点：边界节点 + 向内扩展至多层“内圈”（如启用）
            candidates = set(region_to_boundary.get(r, []))
            if consider_interior_ring and interior_ring_depth > 0:
                frontier = list(candidates)
                visited = set(candidates)
                depth = 0
                while depth < interior_ring_depth and frontier:
                    next_frontier = []
                    for b in frontier:
                        for nb in neighbors[b]:
                            if membership[nb] == r and nb not in visited:
                                visited.add(nb)
                                next_frontier.append(nb)
                    frontier = next_frontier
                    depth += 1
                candidates = visited
            cand_sorted = sorted(candidates, key=lambda i: -(node_O[i] + node_D[i]))

            for i in cand_sorted:
                if membership[i] != r:
                    continue
                best_delta = 0.0
                best_target_region = None
                oi, di = node_O[i], node_D[i]
                # 目标区域：1跳或（必要时）2跳可达的不同区域；如允许全局，则加入全局缺口区
                target_regions = set(node_to_neighbor_regions_1hop[i])
                if not target_regions and max_hops_for_target >= 2:
                    for j in neighbors[i]:
                        target_regions.update(node_to_neighbor_regions_1hop[j])
                if allow_global_target:
                    target_regions.update(top_deficit_regions)
                for s in target_regions:
                    sidx = region_index[s]
                    # 仅向 O/D 均低于目标的区域转移更有意义
                    if O[sidx] >= Ot[sidx] + tol_O and D[sidx] >= Dt[sidx] + tol_D:
                        continue
                    # 估计移动后的 SSE 改善量
                    O_r_old, D_r_old = O[ridx], D[ridx]
                    O_s_old, D_s_old = O[sidx], D[sidx]
                    O_r_new, D_r_new = O_r_old - oi, D_r_old - di
                    O_s_new, D_s_new = O_s_old + oi, D_s_old + di
                    old_pair = (O_r_old - Ot[ridx]) ** 2 + (D_r_old - Dt[ridx]) ** 2 \
                               + (O_s_old - Ot[sidx]) ** 2 + (D_s_old - Dt[sidx]) ** 2
                    new_pair = (O_r_new - Ot[ridx]) ** 2 + (D_r_new - Dt[ridx]) ** 2 \
                               + (O_s_new - Ot[sidx]) ** 2 + (D_s_new - Dt[sidx]) ** 2
                    delta = old_pair - new_pair
                    if delta > best_delta:
                        best_delta = delta
                        best_target_region = s
                if best_target_region is not None and best_delta > 0:
                    if preserve_connectivity:
                        # 连通性保护：仅当节点在原区域至少有两个同区邻居时才允许移动
                        same_region_neighbors = 0
                        for j in neighbors[i]:
                            if membership[j] == r:
                                same_region_neighbors += 1
                        if same_region_neighbors < 2:
                            continue
                    # 执行移动
                    s = best_target_region
                    sidx = region_index[s]
                    membership[i] = s
                    O[ridx] -= oi
                    D[ridx] -= di
                    O[sidx] += oi
                    D[sidx] += di
                    current_sse -= best_delta
                    moves += 1
                    if moves >= max_moves_per_pass:
                        break
            if moves >= max_moves_per_pass:
                break
        if log_progress:
            max_dev_O = float(np.max(np.abs(O - Ot))) if region_count > 0 else 0.0
            max_dev_D = float(np.max(np.abs(D - Dt))) if region_count > 0 else 0.0
            print(f"[平衡 {passes}] moves={moves}, sse={current_sse:.1f}, max|O-dev|={max_dev_O:.1f}, max|D-dev|={max_dev_D:.1f}")
        if moves == 0:
            break
    return membership

def split_regions_until_constraints(graph, membership, nodes_coords,
                                    node_origin_counts, node_destination_counts,
                                    od_node_density,
                                    alpha, beta,
                                    max_edges_per_region=5000,
                                    od_balance_tolerance=0.10,
                                    max_split_passes=5,
                                    max_regions=80,
                                    log_progress=True):
    """
    针对过载区域（内部边过多或 O/D 偏差过大）执行区域内 Leiden 细分，直到所有区域满足约束或达到上限。
    """
    membership = list(membership)

    def compute_stats(mb):
        regions = sorted(set(mb))
        region_nodes = {r: [] for r in regions}
        for idx, label in enumerate(mb):
            region_nodes[label].append(idx)
        stats = {r: {
            'nodes': nodes,
            'O': 0.0,
            'D': 0.0,
            'edge_count_internal': 0
        } for r, nodes in region_nodes.items()}
        # O/D
        for r, nodes in region_nodes.items():
            o_sum = 0.0
            d_sum = 0.0
            for vi in nodes:
                nid = graph.vs[vi]['id']
                o_sum += float(node_origin_counts.get(nid, 0.0))
                d_sum += float(node_destination_counts.get(nid, 0.0))
            stats[r]['O'] = o_sum
            stats[r]['D'] = d_sum
        # 内部边数
        for e in graph.es:
            u, v = e.source, e.target
            ru, rv = mb[u], mb[v]
            if ru == rv:
                stats[ru]['edge_count_internal'] += 1
        return stats

    def od_targets(stats):
        regions = list(stats.keys())
        O_total = sum(stats[r]['O'] for r in regions)
        D_total = sum(stats[r]['D'] for r in regions)
        n = max(1, len(regions))
        Ot = O_total / n
        Dt = D_total / n
        return Ot, Dt

    def needs_split(r, s, Ot, Dt):
        over_edges = s['edge_count_internal'] > max_edges_per_region
        over_od = (abs(s['O'] - Ot) > od_balance_tolerance * max(1.0, Ot)) or (abs(s['D'] - Dt) > od_balance_tolerance * max(1.0, Dt))
        return over_edges or over_od

    def split_region_once(label):
        # 区域总数不可超过 max_regions
        current_regions = len(set(membership))
        if current_regions >= max_regions:
            return False
        nodes = stats[label]['nodes']
        if len(nodes) < 4:
            return False
        sub = graph.induced_subgraph(nodes)
        # 在子图上运行 Leiden（使用相同 alpha/beta，提升分辨率确保至少 2 簇）
        part = run_leiden_with_escalation(
            sub, nodes_coords, alpha=alpha, beta=beta, od_node_density=od_node_density,
            base_resolution=1.0, min_clusters=2, max_tries=4, factor=1.6
        )
        k = len(part)
        if k < 2:
            return False
        # 只保留前两大簇作为有效拆分目标，其他小簇分配到就近的两大簇以避免过度增区
        current_max_label = max(set(membership))
        sub_membership = part.membership
        sub_label_to_nodes = {}
        for local_idx, sub_label in enumerate(sub_membership):
            sub_label_to_nodes.setdefault(sub_label, []).append(local_idx)
        # 跳过单簇的情况
        if len(sub_label_to_nodes) < 2:
            return False
        # 排序，取前两大簇
        ordered = sorted(sub_label_to_nodes.items(), key=lambda kv: -len(kv[1]))
        keep_two = ordered[:2]
        # 计算两大簇的几何质心
        def centroid_of_local_nodes(local_nodes):
            xs, ys = [], []
            for ln in local_nodes:
                gid = nodes[ln]
                nid = graph.vs[gid]['id']
                c = nodes_coords[nid]
                xs.append(c['x'])
                ys.append(c['y'])
            return (float(np.mean(xs)) if xs else 0.0, float(np.mean(ys)) if ys else 0.0)
        (label_a, nodes_a), (label_b, nodes_b) = keep_two[0], keep_two[1]
        ca = centroid_of_local_nodes(nodes_a)
        cb = centroid_of_local_nodes(nodes_b)

        # 将其他小簇分配到最近的两大簇
        for sub_label, local_nodes in ordered[2:]:
            for ln in local_nodes:
                gid = nodes[ln]
                nid = graph.vs[gid]['id']
                c = nodes_coords[nid]
                da = np.hypot(c['x'] - ca[0], c['y'] - ca[1])
                db = np.hypot(c['x'] - cb[0], c['y'] - cb[1])
                if da <= db:
                    nodes_a.append(ln)
                else:
                    nodes_b.append(ln)

        # 应用拆分：第一大簇保留原标签，第二大簇新建标签
        keep_label = label
        new_label_counter = current_max_label + 1
        # 更新全局 membership（仅两类）
        for ln in nodes_a:
            gid = nodes[ln]
            membership[gid] = keep_label
        # 检查是否还能增加新区域（不超过 max_regions）
        if len(set(membership)) >= max_regions:
            return True
        new_label = new_label_counter
        for ln in nodes_b:
            gid = nodes[ln]
            membership[gid] = new_label
        return True

    passes = 0
    while passes < max_split_passes:
        passes += 1
        stats = compute_stats(membership)
        Ot, Dt = od_targets(stats)
        # 找最需要拆分的区域（按内部边超额比例和 O/D 偏差排序）
        candidates = []
        for r, s in stats.items():
            edge_over = max(0, s['edge_count_internal'] - max_edges_per_region)
            od_dev = abs(s['O'] - Ot) + abs(s['D'] - Dt)
            if needs_split(r, s, Ot, Dt):
                score = (edge_over, od_dev, s['edge_count_internal'])
                candidates.append((score, r))
        if not candidates:
            if log_progress:
                print(f"[拆分] 所有区域满足约束，结束。")
            break
        # 逐个尝试拆分若干个候选（一次至少成功一个）
        candidates.sort(key=lambda x: (-x[0][0], -x[0][1], -x[0][2]))
        did_split = False
        for _, r in candidates[:5]:
            if split_region_once(r):
                did_split = True
                if log_progress:
                    print(f"[拆分] 区域 {r} 已细分。")
                break
        if not did_split:
            if log_progress:
                print(f"[拆分] 候选均无法有效细分，结束。")
            break
    return membership

def enforce_region_limits(graph, membership, nodes_coords,
                          node_origin_counts, node_destination_counts,
                          max_regions=80, min_edges_per_region=100,
                          log_progress=True):
    """
    通过合并保证：区域数 ≤ max_regions，且每区内部边 ≥ min_edges_per_region。
    合并选择尽量保持 O/D 均衡并使边规模接近目标。
    """
    membership = list(membership)

    def relabel(mb):
        labels = sorted(set(mb))
        mapping = {old: i for i, old in enumerate(labels)}
        return [mapping[x] for x in mb]

    def compute_stats_and_adj(mb):
        regions = sorted(set(mb))
        region_nodes = {r: [] for r in regions}
        for i, r in enumerate(mb):
            region_nodes[r].append(i)
        stats = {r: {'nodes': ns, 'O': 0.0, 'D': 0.0, 'E': 0, 'centroid': (0.0, 0.0)} for r, ns in region_nodes.items()}
        for r, ns in region_nodes.items():
            xs, ys = [], []
            o, d = 0.0, 0.0
            for vi in ns:
                nid = graph.vs[vi]['id']
                c = nodes_coords[nid]
                xs.append(c['x'])
                ys.append(c['y'])
                o += float(node_origin_counts.get(nid, 0.0))
                d += float(node_destination_counts.get(nid, 0.0))
            stats[r]['O'] = o
            stats[r]['D'] = d
            stats[r]['centroid'] = (float(np.mean(xs)) if xs else 0.0, float(np.mean(ys)) if ys else 0.0)
        adj = defaultdict(set)
        for e in graph.es:
            u, v = e.source, e.target
            ru, rv = mb[u], mb[v]
            if ru == rv:
                stats[ru]['E'] += 1
            else:
                adj[ru].add(rv)
                adj[rv].add(ru)
        return stats, adj

    iterations = 0
    while iterations < 400:
        iterations += 1
        stats, adj = compute_stats_and_adj(membership)
        regions = sorted(stats.keys())
        k = len(regions)
        total_E = sum(stats[r]['E'] for r in regions)
        target_E = total_E / max(1, min(k, max_regions))
        O_total = sum(stats[r]['O'] for r in regions)
        D_total = sum(stats[r]['D'] for r in regions)
        Ot = O_total / max(1, k)
        Dt = D_total / max(1, k)

        # 先合并边过小的区域
        small_regions = [r for r in regions if stats[r]['E'] < min_edges_per_region]
        if small_regions:
            r = min(small_regions, key=lambda x: stats[x]['E'])
            # 邻居优先，否则选最近质心
            neigh = list(adj.get(r, []))
            if not neigh:
                # 选择非自身的最近质心区域
                rx, ry = stats[r]['centroid']
                neigh = sorted([s for s in regions if s != r], key=lambda s: np.hypot(rx - stats[s]['centroid'][0], ry - stats[s]['centroid'][1]))
            best_s, best_score = None, -1e18
            for s in neigh:
                merged_E = stats[r]['E'] + stats[s]['E']
                rx, ry = stats[r]['centroid']
                sx, sy = stats[s]['centroid']
                dist = np.hypot(rx - sx, ry - sy) + 1e-6
                # OD 改善（基于当前均值近似）
                Odiff = (abs(stats[r]['O'] - Ot) + abs(stats[r]['D'] - Dt) + abs(stats[s]['O'] - Ot) + abs(stats[s]['D'] - Dt)) \
                        - (abs(stats[r]['O'] + stats[s]['O'] - Ot) + abs(stats[r]['D'] + stats[s]['D'] - Dt))
                size_closeness = -abs(merged_E - target_E) / (target_E + 1e-6)
                score = 2.0 * size_closeness + 1.0 * Odiff - 0.01 * dist
                if score > best_score:
                    best_score = score
                    best_s = s
            if best_s is None:
                break
            # 执行合并 best_s -> r
            for i in range(len(membership)):
                if membership[i] == best_s:
                    membership[i] = r
            membership = relabel(membership)
            if log_progress:
                print(f"[约束] 合并小边区 {r} <- {best_s}")
            continue

        # 其次控制区域数不超过 max_regions
        if k > max_regions:
            # 从所有相邻对中选择最佳合并对
            best_pair = None
            best_score = -1e18
            for r in regions:
                for s in adj.get(r, []):
                    if r >= s:
                        continue
                    merged_E = stats[r]['E'] + stats[s]['E']
                    rx, ry = stats[r]['centroid']
                    sx, sy = stats[s]['centroid']
                    dist = np.hypot(rx - sx, ry - sy) + 1e-6
                    Odiff = (abs(stats[r]['O'] - Ot) + abs(stats[r]['D'] - Dt) + abs(stats[s]['O'] - Ot) + abs(stats[s]['D'] - Dt)) \
                            - (abs(stats[r]['O'] + stats[s]['O'] - Ot) + abs(stats[r]['D'] + stats[s]['D'] - Dt))
                    size_closeness = -abs(merged_E - target_E) / (target_E + 1e-6)
                    score = 2.0 * size_closeness + 1.5 * Odiff - 0.01 * dist
                    if score > best_score:
                        best_score = score
                        best_pair = (r, s)
            if best_pair is None:
                # 若没有相邻对，选最近质心两个合并
                best_pair = None
                best_score = 1e18
                for i, r in enumerate(regions):
                    rx, ry = stats[r]['centroid']
                    for s in regions[i+1:]:
                        sx, sy = stats[s]['centroid']
                        dist = np.hypot(rx - sx, ry - sy)
                        if dist < best_score:
                            best_score = dist
                            best_pair = (r, s)
            r, s = best_pair
            for i in range(len(membership)):
                if membership[i] == s:
                    membership[i] = r
            membership = relabel(membership)
            if log_progress:
                print(f"[约束] 限制区域数，合并 {r} <- {s}")
            continue

        # 满足约束
        break

    return membership

def enforce_od_range(graph, membership, nodes_coords,
                     node_origin_counts, node_destination_counts,
                     od_min_ratio=0.6, od_max_ratio=1.6,
                     abs_min_od=None, abs_max_od=None,
                     min_edges_per_region=100,
                     max_regions=80,
                     min_regions=None,
                     max_iterations=40,
                     log_progress=True):
    """
    将每个区域的 O+D 总量限制在范围内：
    - 若提供 abs_min_od/abs_max_od，则采用绝对区间 [abs_min_od, abs_max_od]
    - 否则采用相对区间 [od_min_ratio, od_max_ratio] × 当前均值
    - 对过小区域：优先与相邻的低 O+D 区域合并（接近目标），但不使区域数低于下限
    - 对过大区域：在子图内进行二分拆分（不超过 max_regions），必要时优先拆分以提高区域数
    同时保证合并后/拆分后区域内部边 ≥ min_edges_per_region。
    """
    membership = list(membership)

    def compute_stats_and_adj(mb):
        regions = sorted(set(mb))
        region_nodes = {r: [] for r in regions}
        for i, r in enumerate(mb):
            region_nodes[r].append(i)
        stats = {r: {'nodes': ns, 'O': 0.0, 'D': 0.0, 'E': 0, 'centroid': (0.0, 0.0)} for r, ns in region_nodes.items()}
        for r, ns in region_nodes.items():
            xs, ys = [], []
            o, d = 0.0, 0.0
            for vi in ns:
                nid = graph.vs[vi]['id']
                c = nodes_coords[nid]
                xs.append(c['x'])
                ys.append(c['y'])
                o += float(node_origin_counts.get(nid, 0.0))
                d += float(node_destination_counts.get(nid, 0.0))
            stats[r]['O'] = o
            stats[r]['D'] = d
            stats[r]['centroid'] = (float(np.mean(xs)) if xs else 0.0, float(np.mean(ys)) if ys else 0.0)
        adj = defaultdict(set)
        for e in graph.es:
            u, v = e.source, e.target
            ru, rv = mb[u], mb[v]
            if ru == rv:
                stats[ru]['E'] += 1
            else:
                adj[ru].add(rv)
                adj[rv].add(ru)
        return stats, adj

    def relabel(mb):
        labels = sorted(set(mb))
        mapping = {old: i for i, old in enumerate(labels)}
        return [mapping[x] for x in mb]

    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        stats, adj = compute_stats_and_adj(membership)
        regions = sorted(stats.keys())
        k = len(regions)
        total_od = sum(stats[r]['O'] + stats[r]['D'] for r in regions)
        if k == 0:
            break
        # 绝对或相对上下界
        use_abs = (abs_min_od is not None and abs_max_od is not None)
        if use_abs:
            lower = float(abs_min_od)
            upper = float(abs_max_od)
            target = 0.5 * (lower + upper)
            k_min_needed = int(np.ceil(total_od / max(upper, 1e-6)))
        else:
            target = total_od / float(k)
            lower = od_min_ratio * target
            upper = od_max_ratio * target
            k_min_needed = 0
        k_min_limit = max(min_regions or 0, k_min_needed)

        # 标记过小/过大
        small = [r for r in regions if (stats[r]['O'] + stats[r]['D']) < lower]
        big = [r for r in regions if (stats[r]['O'] + stats[r]['D']) > upper]

        changed = False

        # 若采用绝对区间且区域数不足下限，优先拆分最大的超标大区以提升区域数
        if use_abs and len(set(membership)) <= k_min_limit and big and len(set(membership)) < max_regions:
            r_big = max(big, key=lambda x: (stats[x]['O'] + stats[x]['D']))
            nodes = stats[r_big]['nodes']
            if len(nodes) >= 4:
                sub = graph.induced_subgraph(nodes)
                sub_od = {}
                vals = []
                for li, gid in enumerate(nodes):
                    nid = graph.vs[gid]['id']
                    v = float(node_origin_counts.get(nid, 0.0)) + float(node_destination_counts.get(nid, 0.0))
                    sub_od[nid] = v
                    vals.append(v)
                if vals:
                    vmin, vmax = float(min(vals)), float(max(vals))
                    if vmax > vmin:
                        for nid in list(sub_od.keys()):
                            sub_od[nid] = (sub_od[nid] - vmin) / (vmax - vmin + 1e-6)
                    else:
                        for nid in list(sub_od.keys()):
                            sub_od[nid] = 0.0
                part = run_leiden_with_escalation(
                    sub, nodes_coords, alpha=0.2, beta=0.8, od_node_density=sub_od,
                    base_resolution=1.0, min_clusters=2, max_tries=4, factor=1.8
                )
                if len(part) >= 2:
                    comp = {}
                    for li, lab in enumerate(part.membership):
                        comp.setdefault(lab, []).append(li)
                    ordered = sorted(comp.items(), key=lambda kv: -len(kv[1]))[:2]
                    a, b = ordered[0][1], ordered[1][1]
                    current_max = max(set(membership))
                    new_label = current_max + 1
                    for ln in a:
                        gid = nodes[ln]
                        membership[gid] = r_big
                    for ln in b:
                        gid = nodes[ln]
                        membership[gid] = new_label
                    membership = relabel(membership)
                    changed = True
                    if log_progress:
                        print(f"[OD范围-绝对] 区域数不足，先拆分大区 {r_big} 以提升区域数")
        if changed:
            continue

        # 合并小区域，优先相邻低OD且合并后不低于 min_edges_per_region；绝对区间下若达下限则不再合并
        for r in list(small):
            if use_abs and len(set(membership)) <= k_min_limit:
                break
            neigh = list(adj.get(r, []))
            if not neigh:
                # 选最近质心区域
                rx, ry = stats[r]['centroid']
                neigh = sorted([s for s in regions if s != r], key=lambda s: np.hypot(rx - stats[s]['centroid'][0], ry - stats[s]['centroid'][1]))
            best_s, best_score = None, 1e18
            for s in neigh:
                merged_od = (stats[r]['O'] + stats[r]['D']) + (stats[s]['O'] + stats[s]['D'])
                merged_E = stats[r]['E'] + stats[s]['E']
                if merged_E < min_edges_per_region:
                    continue
                if use_abs:
                    if merged_od <= upper:
                        score = abs(merged_od - target)
                    else:
                        score = abs(merged_od - upper) + 1e6
                else:
                    score = abs(merged_od - target)
                if score < best_score:
                    best_score = score
                    best_s = s
            if best_s is not None:
                # 执行合并：best_s -> r
                for i in range(len(membership)):
                    if membership[i] == best_s:
                        membership[i] = r
                membership = relabel(membership)
                changed = True
        if changed:
            continue

        # 若区域数已达上限且仍存在超大OD区域：先合并两个小OD相邻区域以释放名额
        if len(set(membership)) >= max_regions and big:
            regs_sorted = sorted(regions, key=lambda r: (stats[r]['O'] + stats[r]['D']))
            best_pair = None
            best_score = 1e18
            for r in regs_sorted:
                # 按相邻低OD优先
                neigh = sorted(list(adj.get(r, [])), key=lambda s: (stats[s]['O'] + stats[s]['D']))
                for s in neigh:
                    if r >= s:
                        continue
                    merged_E = stats[r]['E'] + stats[s]['E']
                    if merged_E < min_edges_per_region:
                        continue
                    score = (stats[r]['O'] + stats[r]['D']) + (stats[s]['O'] + stats[s]['D'])
                    if score < best_score:
                        best_score = score
                        best_pair = (r, s)
                if best_pair is not None:
                    break
            # 若没有相邻对，使用最近质心对
            if best_pair is None and regs_sorted:
                best_dist = 1e18
                for i, r in enumerate(regs_sorted):
                    rx, ry = stats[r]['centroid']
                    for s in regs_sorted[i+1:]:
                        if r == s:
                            continue
                        sx, sy = stats[s]['centroid']
                        d = np.hypot(rx - sx, ry - sy)
                        if d < best_dist:
                            best_dist = d
                            best_pair = (r, s)
            if best_pair is not None:
                r, s = best_pair
                for i in range(len(membership)):
                    if membership[i] == s:
                        membership[i] = r
                membership = relabel(membership)
                if log_progress:
                    print(f"[OD范围] 达到上限，先合并 {r} <- {s} 释放名额")
                changed = True
                continue

        # 再拆分大区域（不超过 max_regions）
        if len(set(membership)) < max_regions:
            for r in list(big):
                # 子图二分
                nodes = stats[r]['nodes']
                if len(nodes) < 4:
                    continue
                sub = graph.induced_subgraph(nodes)
                # 基于子图节点的 O+D 构建 OD 密度，以驱动按照 OD 进行切分
                sub_od = {}
                vals = []
                for li, gid in enumerate(nodes):
                    nid = graph.vs[gid]['id']
                    v = float(node_origin_counts.get(nid, 0.0)) + float(node_destination_counts.get(nid, 0.0))
                    sub_od[nid] = v
                    vals.append(v)
                if vals:
                    vmin, vmax = float(min(vals)), float(max(vals))
                    if vmax > vmin:
                        for nid in list(sub_od.keys()):
                            sub_od[nid] = (sub_od[nid] - vmin) / (vmax - vmin + 1e-6)
                    else:
                        for nid in list(sub_od.keys()):
                            sub_od[nid] = 0.0
                # 以 OD 为主导进行二分（beta 高、alpha 低）
                part = run_leiden_with_escalation(
                    sub, nodes_coords, alpha=0.2, beta=0.8, od_node_density=sub_od,
                    base_resolution=1.0, min_clusters=2, max_tries=4, factor=1.8
                )
                ksub = len(part)
                if ksub < 2:
                    continue
                # 仅保留两大簇
                sub_m = part.membership
                comp = {}
                for li, lab in enumerate(sub_m):
                    comp.setdefault(lab, []).append(li)
                ordered = sorted(comp.items(), key=lambda kv: -len(kv[1]))[:2]
                if len(ordered) < 2:
                    continue
                a, b = ordered[0][1], ordered[1][1]
                # 新标签
                current_max = max(set(membership))
                new_label = current_max + 1
                # 应用：a 保持 r，b 改为 new_label
                for ln in a:
                    gid = nodes[ln]
                    membership[gid] = r
                for ln in b:
                    gid = nodes[ln]
                    membership[gid] = new_label
                membership = relabel(membership)
                changed = True
                if len(set(membership)) >= max_regions:
                    break

        if not changed:
            break

    return membership
def refine_partition_with_capacity_label_propagation(
        graph,
        membership,
        node_origin_counts,
        node_destination_counts,
        max_regions=100,
        min_edges_per_region=1000,
        od_weight=1.1,
        edge_weight=0.58,
        smooth_weight=0.08,
        max_rounds=8,
        max_moves_per_round=200000,
        preserve_connectivity=True,
        log_progress=True):
    """
    容量约束的标签传播细化（借鉴 SCLaP/平衡划分思想）：
    - 目标：最小化 O/D 偏差平方和与内部边规模偏差平方和，并惩罚割边增长
    - 硬约束：区域数不增加；任何移动不使区域内部边 < min_edges_per_region
    - 仅考虑边界节点的移动，逐轮贪心迭代
    """
    n = graph.vcount()
    membership = list(membership)

    # 邻接列表
    nbrs = [[] for _ in range(n)]
    for e in graph.es:
        u, v = e.source, e.target
        nbrs[u].append(v)
        nbrs[v].append(u)

    # 节点 O/D
    node_O = np.zeros(n, dtype=float)
    node_D = np.zeros(n, dtype=float)
    for i in range(n):
        nid = graph.vs[i]['id']
        node_O[i] = float(node_origin_counts.get(nid, 0.0))
        node_D[i] = float(node_destination_counts.get(nid, 0.0))

    def compute_stats(mb):
        regs = sorted(set(mb))
        idx = {r: i for i, r in enumerate(regs)}
        k = len(regs)
        O = np.zeros(k, dtype=float)
        D = np.zeros(k, dtype=float)
        E = np.zeros(k, dtype=float)
        N = np.zeros(k, dtype=int)
        for i, r in enumerate(mb):
            ridx = idx[r]
            O[ridx] += node_O[i]
            D[ridx] += node_D[i]
            N[ridx] += 1
        for e in graph.es:
            u, v = e.source, e.target
            ru, rv = mb[u], mb[v]
            if ru == rv:
                E[idx[ru]] += 1
        return regs, idx, O, D, E, N

    def boundary_nodes(mb):
        bset = set()
        for i in range(n):
            ri = mb[i]
            for j in nbrs[i]:
                if mb[j] != ri:
                    bset.add(i)
                    break
        return list(bset)

    def region_neighbors_of_node(i, mb):
        rset = set()
        for j in nbrs[i]:
            rset.add(mb[j])
        rset.discard(mb[i])
        return rset

    def neighbors_in_region(i, region_label, mb):
        c = 0
        for j in nbrs[i]:
            if mb[j] == region_label:
                c += 1
        return c

    regs, ridx, O, D, E, N = compute_stats(membership)
    k_eff = min(len(regs), max_regions)
    Ot = float(np.sum(O)) / max(1, k_eff)
    Dt = float(np.sum(D)) / max(1, k_eff)
    Et = float(np.sum(E)) / max(1, k_eff)

    def move_cost_delta(i, r_from, r_to):
        rf, rt = ridx[r_from], ridx[r_to]
        oi, di = node_O[i], node_D[i]
        # 邻接内部边变化估计
        nin_from = neighbors_in_region(i, r_from, membership)
        nin_to = neighbors_in_region(i, r_to, membership)
        E_from_new = E[rf] - nin_from
        E_to_new = E[rt] + nin_to
        # 硬约束：源区域内部边不能降到阈值以下
        if E_from_new < min_edges_per_region:
            return None
        # O/D 偏差平方和变化
        def sq(x):
            return x * x
        od_before = sq(O[rf] - Ot) + sq(D[rf] - Dt) + sq(O[rt] - Ot) + sq(D[rt] - Dt)
        od_after = sq(O[rf] - oi - Ot) + sq(D[rf] - di - Dt) + sq(O[rt] + oi - Ot) + sq(D[rt] + di - Dt)
        delta_od = od_after - od_before
        # E 偏差平方和变化
        e_before = sq(E[rf] - Et) + sq(E[rt] - Et)
        e_after = sq(E_from_new - Et) + sq(E_to_new - Et)
        delta_e = e_after - e_before
        # 平滑/割边：倾向于 nin_to 大、nin_from 小
        cut_delta = (nin_from - nin_to)
        total_delta = od_weight * delta_od + edge_weight * delta_e + smooth_weight * cut_delta
        return total_delta, nin_from, nin_to

    rounds = 0
    while rounds < max_rounds:
        rounds += 1
        moves = 0
        bnodes = boundary_nodes(membership)
        # 贪心：按节点 (O+D) 权重从大到小
        bnodes.sort(key=lambda i: -(node_O[i] + node_D[i]))
        for i in bnodes:
            r_from = membership[i]
            cand_regions = region_neighbors_of_node(i, membership)
            best = None
            for r_to in cand_regions:
                res = move_cost_delta(i, r_from, r_to)
                if res is None:
                    continue
                delta, nin_from, nin_to = res
                if best is None or delta < best[0]:
                    best = (delta, r_to, nin_from, nin_to)
            if best is not None and best[0] < -1e-9:
                if preserve_connectivity:
                    # 连通性保护：在原区域至少保留两个同区邻居
                    same_region_neighbors = 0
                    for j in nbrs[i]:
                        if membership[j] == r_from:
                            same_region_neighbors += 1
                    if same_region_neighbors < 2:
                        continue
                # 应用移动
                _, r_to, nin_from, nin_to = best
                rf, rt = ridx[r_from], ridx[r_to]
                oi, di = node_O[i], node_D[i]
                # 更新统计
                O[rf] -= oi; D[rf] -= di; E[rf] -= nin_from
                O[rt] += oi; D[rt] += di; E[rt] += nin_to
                membership[i] = r_to
                moves += 1
                if moves >= max_moves_per_round:
                    break
        if log_progress:
            print(f"[容量细化 {rounds}] moves={moves}, max|O-dev|={float(np.max(np.abs(O - Ot))):.1f}, max|D-dev|={float(np.max(np.abs(D - Dt))):.1f}, max|E-dev|={float(np.max(np.abs(E - Et))):.1f}")
        if moves == 0:
            break
        # 轮末重计算（标签映射不变）
        regs, ridx, O, D, E, N = compute_stats(membership)
        k_eff = min(len(regs), max_regions)
        Ot = float(np.sum(O)) / max(1, k_eff)
        Dt = float(np.sum(D)) / max(1, k_eff)
        Et = float(np.sum(E)) / max(1, k_eff)

    return membership

def spatial_majority_smooth_labels(
        graph,
        membership,
        node_origin_counts,
        node_destination_counts,
        majority_ratio=0.6,
        max_rounds=2,
        only_boundary=True,
        od_low_quantile=0.7,
        log_progress=True):
    """
    多数投票空间平滑：
    - 仅对边界节点（或全部节点）执行：若邻居中某区域标签占比≥majority_ratio，则将该节点改为该标签；
    - 优先搬运 O+D 较小的节点（阈值=全局 O+D 的 od_low_quantile 分位数），降低对 O/D 约束的冲击；
    - 连续执行 max_rounds 直到收敛或达到上限。
    """
    n = graph.vcount()
    membership = list(membership)
    # 邻接
    nbrs = [[] for _ in range(n)]
    for e in graph.es:
        u, v = e.source, e.target
        nbrs[u].append(v)
        nbrs[v].append(u)
    # 节点 O+D
    node_w = np.zeros(n, dtype=float)
    for i in range(n):
        nid = graph.vs[i]['id']
        node_w[i] = float(node_origin_counts.get(nid, 0.0)) + float(node_destination_counts.get(nid, 0.0))
    # 分位阈值（若全 0 则阈值 0）
    try:
        thr = float(np.quantile(node_w, od_low_quantile)) if np.any(node_w > 0) else 0.0
    except Exception:
        thr = float(np.percentile(node_w, od_low_quantile * 100)) if np.any(node_w > 0) else 0.0

    rounds = 0
    while rounds < max_rounds:
        rounds += 1
        moves = 0
        # 候选节点
        if only_boundary:
            candidates = []
            for i in range(n):
                ri = membership[i]
                for j in nbrs[i]:
                    if membership[j] != ri:
                        candidates.append(i)
                        break
        else:
            candidates = list(range(n))

        for i in candidates:
            deg = len(nbrs[i])
            if deg == 0:
                continue
            # 仅搬运低 O+D 节点
            if node_w[i] > thr:
                continue
            counts = {}
            for j in nbrs[i]:
                rj = membership[j]
                counts[rj] = counts.get(rj, 0) + 1
            # 多数标签
            best_region, best_cnt = None, -1
            for rj, c in counts.items():
                if c > best_cnt:
                    best_cnt = c
                    best_region = rj
            if best_region is None or best_region == membership[i]:
                continue
            if best_cnt >= majority_ratio * float(deg):
                membership[i] = best_region
                moves += 1
        if log_progress:
            print(f"[空间平滑 {rounds}] moves={moves}, candidates={len(candidates)}")
        if moves == 0:
            break

    return membership

def absorb_small_islands(
        graph,
        membership,
        min_nodes=50,
        min_internal_edges=80,
        max_passes=2,
        log_progress=True):
    """
    吸收小型“岛屿”分量：
    - 对每个标签，在其诱导子图中分解连通分量；
    - 若分量节点数 < min_nodes 或内部边 < min_internal_edges，则将整个分量并入与其边界接触最多的相邻标签；
    - 重复若干轮直至无变化或达上限。
    """
    membership = list(membership)

    def relabel(mb):
        labels = sorted(set(mb))
        mapping = {old: i for i, old in enumerate(labels)}
        return [mapping[x] for x in mb]

    passes = 0
    while passes < max_passes:
        passes += 1
        changed = False
        label_to_nodes = {}
        for i, r in enumerate(membership):
            label_to_nodes.setdefault(r, []).append(i)
        for r, nodes in label_to_nodes.items():
            if len(nodes) <= 1:
                continue
            sub = graph.induced_subgraph(nodes)
            comps = sub.components().membership
            comp_to_local_nodes = {}
            for li, cid in enumerate(comps):
                comp_to_local_nodes.setdefault(cid, []).append(li)
            if len(comp_to_local_nodes) <= 1:
                continue
            for cid, local_list in comp_to_local_nodes.items():
                comp_global_nodes = [nodes[li] for li in local_list]
                # 统计内部边（在子图内按分量计数）
                comp_idx = set(local_list)
                internal_edges = 0
                for e in sub.es:
                    if e.source in comp_idx and e.target in comp_idx:
                        internal_edges += 1
                if len(comp_global_nodes) >= min_nodes and internal_edges >= min_internal_edges:
                    continue
                # 统计与外部标签的接触次数
                touch = {}
                for gi in comp_global_nodes:
                    for e in graph.incident(gi):
                        u, v = graph.es[e].source, graph.es[e].target
                        nb = v if u == gi else u
                        tr = membership[nb]
                        if tr == r:
                            continue
                        touch[tr] = touch.get(tr, 0) + 1
                if not touch:
                    continue
                tgt = max(touch.keys(), key=lambda x: touch[x])
                for gi in comp_global_nodes:
                    membership[gi] = tgt
                changed = True
        if log_progress:
            print(f"[吸收小岛 {passes}] changed={changed}")
        if not changed:
            break
        membership = relabel(membership)

    return membership

def generate_edge_mappings_and_adjacency(sumo_edges, node_to_region, alpha):
    """
    基于节点聚类结果生成边到区域的映射和边邻接关系
    
    Args:
        sumo_edges: SUMO边信息字典 {edge_id: {from_node, to_node, ...}}
        node_to_region: 节点到区域的映射 {node_id: region_id}
        alpha: alpha参数，用于文件命名
    """
    print("\n" + "="*50)
    print("开始生成边-区域映射和边邻接关系...")
    start_time = time.time()
    
    # 1. 生成边到区域的映射（仅统计“内部边”），跨区域边只作为边界边记录
    edge_to_region = {}
    internal_edges = 0
    cross_region_edges = 0
    unmapped_edges = 0
    
    for edge_id, edge_info in sumo_edges.items():
        from_node = edge_info['from_node']
        to_node = edge_info['to_node']
        
        from_region = node_to_region.get(from_node)
        to_region = node_to_region.get(to_node)
        
        if from_region is not None and to_region is not None:
            if from_region == to_region:
                edge_to_region[edge_id] = from_region
                internal_edges += 1
            else:
                # 跨区域边不计入任一区域的内部边，作为边界边单独记录
                cross_region_edges += 1
        else:
            unmapped_edges += 1
    
    print(f"边-区域映射完成:")
    print(f"  内部边数量: {internal_edges}")
    print(f"  跨区域边数量: {cross_region_edges}")
    print(f"  无法映射的边数: {unmapped_edges}")
    
    # 2. 生成边与边的邻接关系
    edge_adjacency = {}
    
    # 建立节点到边的映射（哪些边连接到每个节点）
    node_to_edges = {}
    for edge_id, edge_info in sumo_edges.items():
        from_node = edge_info['from_node']
        to_node = edge_info['to_node']
        
        if from_node not in node_to_edges:
            node_to_edges[from_node] = []
        if to_node not in node_to_edges:
            node_to_edges[to_node] = []
        
        node_to_edges[from_node].append(edge_id)
        node_to_edges[to_node].append(edge_id)
    
    # 生成边邻接关系：如果两条边共享节点，则它们相邻
    for edge_id in sumo_edges:
        edge_adjacency[edge_id] = []
    
    for node_id, connected_edges in node_to_edges.items():
        # 对于每个节点，连接到它的所有边都是相邻的
        for i, edge1 in enumerate(connected_edges):
            for j, edge2 in enumerate(connected_edges):
                if i != j and edge2 not in edge_adjacency[edge1]:
                    edge_adjacency[edge1].append(edge2)
    
    # 统计邻接关系
    total_adjacency_count = sum(len(neighbors) for neighbors in edge_adjacency.values())
    avg_neighbors = total_adjacency_count / len(edge_adjacency) if edge_adjacency else 0
    
    print(f"边邻接关系生成完成:")
    print(f"  总边数: {len(edge_adjacency)}")
    print(f"  总邻接关系数: {total_adjacency_count}")
    print(f"  平均每条边的邻居数: {avg_neighbors:.2f}")
    
    # 3. 保存结果
    # 保存边-区域映射
    edge_mapping_file = f'edge_to_region_alpha_{alpha}.json'
    with open(edge_mapping_file, 'w') as f:
        json.dump(edge_to_region, f, indent=4)
    print(f"边-区域映射已保存到: {edge_mapping_file}")
    
    # 保存边邻接关系
    adjacency_file = f'edge_adjacency_alpha_{alpha}.json'
    with open(adjacency_file, 'w') as f:
        json.dump(edge_adjacency, f, indent=4)
    print(f"边邻接关系已保存到: {adjacency_file}")
    
    # 4. 生成统计报告
    region_edge_counts = {}
    for edge_id, region_id in edge_to_region.items():
        if region_id not in region_edge_counts:
            region_edge_counts[region_id] = 0
        region_edge_counts[region_id] += 1
    
    print(f"\n各区域边数统计:")
    for region_id in sorted(region_edge_counts.keys()):
        count = region_edge_counts[region_id]
        print(f"  区域 {region_id}: {count} 条边")
    
    # 检查跨区域边（边界边）：遍历所有能映射到区域的边
    boundary_edges = []
    for edge_id, edge_info in sumo_edges.items():
        from_region = node_to_region.get(edge_info['from_node'])
        to_region = node_to_region.get(edge_info['to_node'])
        if from_region is not None and to_region is not None and from_region != to_region:
            boundary_edges.append({
                'edge_id': edge_id,
                'from_region': from_region,
                'to_region': to_region
            })
    
    print(f"\n跨区域边界边统计:")
    print(f"  边界边数量: {len(boundary_edges)}")
    
    # 保存边界边信息
    boundary_file = f'boundary_edges_alpha_{alpha}.json'
    with open(boundary_file, 'w') as f:
        json.dump(boundary_edges, f, indent=4)
    print(f"边界边信息已保存到: {boundary_file}")
    
    print(f"边映射和邻接关系生成完成，耗时: {time.time() - start_time:.2f} 秒")
    print("="*50)

def enforce_region_connectivity(graph, membership, log_progress=True):
    """
    将每个区域分解为连通分量，并为非主分量重新指派到邻接区域中与其接触最多的区域。
    保证每个区域在图上是连通的。
    """
    membership = list(membership)
    # 构建区域到节点列表
    region_to_nodes = {}
    for i, r in enumerate(membership):
        region_to_nodes.setdefault(r, []).append(i)
    # 逐区域处理
    for r, nodes in region_to_nodes.items():
        if len(nodes) <= 1:
            continue
        # 提取子图并找连通分量
        sub = graph.induced_subgraph(nodes)
        comps = sub.components().membership
        # 统计每个分量大小
        comp_to_nodes_local = {}
        for li, comp_id in enumerate(comps):
            comp_to_nodes_local.setdefault(comp_id, []).append(li)
        if len(comp_to_nodes_local) <= 1:
            continue
        # 找到主分量（最大）
        main_comp_id = max(comp_to_nodes_local.keys(), key=lambda cid: len(comp_to_nodes_local[cid]))
        main_set_global = set(nodes[li] for li in comp_to_nodes_local[main_comp_id])
        # 处理其余分量：把这些节点分配到与其接触最多的相邻区域
        for cid, local_list in comp_to_nodes_local.items():
            if cid == main_comp_id:
                continue
            for li in local_list:
                gi = nodes[li]
                # 统计邻接区域触点数
                touch = {}
                for e in graph.incident(gi):
                    u, v = graph.es[e].source, graph.es[e].target
                    nb = v if u == gi else u
                    tr = membership[nb]
                    if tr == r:
                        continue
                    touch[tr] = touch.get(tr, 0) + 1
                if not touch:
                    # 若无外部邻接，则分配到主分量区域（保持 r）
                    continue
                tgt = max(touch.keys(), key=lambda x: touch[x])
                membership[gi] = tgt
        if log_progress:
            print(f"[连通性] 区域 {r} 被拆成 {len(comp_to_nodes_local)} 个分量，已重指派小分量")
    return membership

# --- 主函数 ---
if __name__ == '__main__':
    SUMO_NET_FILE = '/data/XXXXX/LLMNavigation/Data/NYC/NewYork.net.xml'
    SUMO_ROU_FILE = '/data/XXXXX/LLMNavigation/Data/NYC/NewYork_od_0.1.rou.alt.xml'
    ALPHA_VALUE = 1
    BETA_VALUE = 2
    RESOLUTION_VALUE = 1.0
    
    try:
        graph, nodes_coords, _, sumo_edges = parse_sumo_net_to_igraph(SUMO_NET_FILE)

        # 解析 OD
        node_origin_counts, node_destination_counts, node_od_density = parse_sumo_routes_od(SUMO_ROU_FILE, sumo_edges)

        # 初次 Leiden 聚类采用“分辨率自适应初始化”（过分割以利后续合并）
        initial_partition = run_leiden_with_escalation(
            graph, nodes_coords,
            alpha=ALPHA_VALUE,
            beta=BETA_VALUE,
            od_node_density=node_od_density,
            base_resolution=RESOLUTION_VALUE,
            min_clusters=max(2 * 50, 100),  # 至少为目标区域数的2倍
            max_tries=6,
            factor=1.5
        )

        # 基于 OD 的迭代合并，满足平衡与区域数/规模约束
        merged_membership = iterative_merge_regions(
            graph,
            initial_partition.membership,
            nodes_coords,
            node_origin_counts,
            node_destination_counts,
            max_regions=100,
            min_regions=60,
            desired_regions=80,
            imbalance_threshold=0.10,
            size_tolerance=0.10,
            max_iterations=200,
            log_progress=True
        )

        # 合并后输出各区域 O/D 与规模统计
        region_od_summary = defaultdict(lambda: {'O': 0.0, 'D': 0.0, 'N': 0})
        for i, label in enumerate(merged_membership):
            nid = graph.vs[i]['id']
            region_od_summary[label]['O'] += float(node_origin_counts.get(nid, 0.0))
            region_od_summary[label]['D'] += float(node_destination_counts.get(nid, 0.0))
            region_od_summary[label]['N'] += 1
        print("\n最终区域 O/D/规模统计（前10个区域按节点数排序）：")
        ordered_regions = sorted(region_od_summary.items(), key=lambda kv: -kv[1]['N'])
        for rid, info in ordered_regions[:10]:
            O, D, N = info['O'], info['D'], info['N']
            imb = abs(O - D) / (O + D + 1e-6)
            print(f"  区域 {rid}: N={N}, O={O:.1f}, D={D:.1f}, imbalance={imb:.3f}")

        # 仅依据低 OD 规则进一步合并到接近目标区域数（允许最终区域数非固定）
        od_merged_membership = merge_regions_by_low_od(
            graph,
            merged_membership,
            nodes_coords,
            node_origin_counts,
            node_destination_counts,
            desired_regions=80,
            max_iterations=200,
            log_progress=True
        )

        # 在非固定区域数量的前提下，进行 O/D 负载均衡的边界节点重分配
        balanced_membership = balance_regions_by_od(
            graph,
            od_merged_membership,
            node_origin_counts,
            node_destination_counts,
            od_balance_tolerance=0.10,
            max_passes=20,
            max_moves_per_pass=20000,
            interior_ring_depth=2,
            log_progress=True
        )

        # 空间多数投票平滑（第一次）：吸收少量异色孤岛，边界更规整
        balanced_membership = spatial_majority_smooth_labels(
            graph,
            balanced_membership,
            node_origin_counts,
            node_destination_counts,
            majority_ratio=0.6,
            max_rounds=2,
            only_boundary=True,
            od_low_quantile=0.7,
            log_progress=True
        )
        # 吸收小型岛屿（第一次）
        balanced_membership = absorb_small_islands(
            graph,
            balanced_membership,
            min_nodes=60,
            min_internal_edges=80,
            max_passes=2,
            log_progress=True
        )

        # 若仍有区域内部边超限或 O/D 偏差过大，对超限区域做细分（Leiden 子图），不固定最终区域数
        constrained_membership = split_regions_until_constraints(
            graph,
            balanced_membership,
            nodes_coords,
            node_origin_counts,
            node_destination_counts,
            od_node_density=node_od_density,
            alpha=ALPHA_VALUE,
            beta=BETA_VALUE,
            max_edges_per_region=5000,
            od_balance_tolerance=0.06,
            max_split_passes=8,
            max_regions=80,
            log_progress=True
        )

        # 施加硬性约束：区域数 ≤ 80 且每区内部边 ≥ 100
        constrained_membership = enforce_region_limits(
            graph,
            constrained_membership,
            nodes_coords,
            node_origin_counts,
            node_destination_counts,
            max_regions=80,
            min_edges_per_region=100,
            log_progress=True
        )

        # 再做一次 O/D 负载均衡以消除合并/细分后的偏差
        constrained_membership = balance_regions_by_od(
            graph,
            constrained_membership,
            node_origin_counts,
            node_destination_counts,
            od_balance_tolerance=0.08,
            max_passes=30,
            max_moves_per_pass=50000,
            consider_interior_ring=True,
            max_hops_for_target=2,
            allow_global_target=True,
            log_progress=True
        )
        # 空间多数投票平滑（第二次）：在连通性与约束后再做收口
        constrained_membership = spatial_majority_smooth_labels(
            graph,
            constrained_membership,
            node_origin_counts,
            node_destination_counts,
            majority_ratio=0.6,
            max_rounds=2,
            only_boundary=True,
            od_low_quantile=0.7,
            log_progress=True
        )
        # 吸收小型岛屿（第二次）
        constrained_membership = absorb_small_islands(
            graph,
            constrained_membership,
            min_nodes=60,
            min_internal_edges=80,
            max_passes=2,
            log_progress=True
        )

        # 容量约束的标签传播细化：同时拉齐 O/D 与内部边规模，保持区域数≤80且每区边≥100
        constrained_membership = refine_partition_with_capacity_label_propagation(
            graph,
            constrained_membership,
            node_origin_counts,
            node_destination_counts,
            max_regions=80,
            min_edges_per_region=100,
            od_weight=1.6,
            edge_weight=0.6,
            smooth_weight=0.05,
            max_rounds=8,
            max_moves_per_round=200000,
            log_progress=True
        )

        # 统计并打印最终 O/D 均衡性与边规模
        final_regions = sorted(set(constrained_membership))
        region_stats = {r: {'O': 0.0, 'D': 0.0, 'N': 0, 'Eint': 0} for r in final_regions}
        for i, r in enumerate(constrained_membership):
            nid = graph.vs[i]['id']
            region_stats[r]['O'] += float(node_origin_counts.get(nid, 0.0))
            region_stats[r]['D'] += float(node_destination_counts.get(nid, 0.0))
            region_stats[r]['N'] += 1
        for e in graph.es:
            u, v = e.source, e.target
            ru, rv = constrained_membership[u], constrained_membership[v]
            if ru == rv:
                region_stats[ru]['Eint'] += 1

        def _cv(vals):
            vals = np.array(vals, dtype=float)
            m = float(np.mean(vals)) if vals.size else 0.0
            s = float(np.std(vals)) if vals.size else 0.0
            return (s / (m + 1e-6)) if m > 0 else 0.0
        def _gini(vals):
            x = np.sort(np.array(vals, dtype=float))
            if x.size == 0:
                return 0.0
            n = x.size
            cumx = np.cumsum(x)
            if cumx[-1] == 0:
                return 0.0
            g = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
            return float(max(0.0, g))

        O_list = [region_stats[r]['O'] for r in final_regions]
        D_list = [region_stats[r]['D'] for r in final_regions]
        E_list = [region_stats[r]['Eint'] for r in final_regions]
        print("\n最终区域分布质量指标：")
        print(f"  O: mean={np.mean(O_list):.1f}, max={np.max(O_list):.1f}, CV={_cv(O_list):.3f}, Gini={_gini(O_list):.3f}")
        print(f"  D: mean={np.mean(D_list):.1f}, max={np.max(D_list):.1f}, CV={_cv(D_list):.3f}, Gini={_gini(D_list):.3f}")
        print(f"  内部边: mean={np.mean(E_list):.1f}, max={np.max(E_list):.0f}, CV={_cv(E_list):.3f}")

        # 强制区域连通性
        constrained_membership = enforce_region_connectivity(graph, constrained_membership, log_progress=True)

        # （移至流程末尾后再输出每区 O/D 与规模明细与摘要保存）

        # 强制 O+D 总量位于范围内：合并过小、拆分过大
        constrained_membership = enforce_od_range(
            graph,
            constrained_membership,
            nodes_coords,
            node_origin_counts,
            node_destination_counts,
            od_min_ratio=0.9,
            od_max_ratio=1.1,
            abs_min_od=2000,
            abs_max_od=9000,
            min_edges_per_region=100,
            max_regions=80,
            min_regions=60,
            max_iterations=200,
            log_progress=True
        )

        # 再次执行硬约束、O/D均衡与连通性，确保范围修正后的质量
        constrained_membership = enforce_region_limits(
            graph,
            constrained_membership,
            nodes_coords,
            node_origin_counts,
            node_destination_counts,
            max_regions=80,
            min_edges_per_region=100,
            log_progress=True
        )
        constrained_membership = balance_regions_by_od(
            graph,
            constrained_membership,
            node_origin_counts,
            node_destination_counts,
            od_balance_tolerance=0.06,
            max_passes=25,
            max_moves_per_pass=60000,
            consider_interior_ring=True,
            interior_ring_depth=2,
            max_hops_for_target=2,
            allow_global_target=True,
            preserve_connectivity=True,
            log_progress=True
        )
        constrained_membership = enforce_region_connectivity(graph, constrained_membership, log_progress=True)

        # 流程全部校正后：输出每区域 O/D 与规模明细，并保存摘要（最终结果）
        final_regions = sorted(set(constrained_membership))
        region_stats2 = {r: {'O': 0.0, 'D': 0.0, 'N': 0, 'Eint': 0} for r in final_regions}
        for i, r in enumerate(constrained_membership):
            nid = graph.vs[i]['id']
            region_stats2[r]['O'] += float(node_origin_counts.get(nid, 0.0))
            region_stats2[r]['D'] += float(node_destination_counts.get(nid, 0.0))
            region_stats2[r]['N'] += 1
        for e in graph.es:
            u, v = e.source, e.target
            ru, rv = constrained_membership[u], constrained_membership[v]
            if ru == rv:
                region_stats2[ru]['Eint'] += 1
        print("\n最终每区域 O/D 与规模统计（按区域ID排序）：")
        region_summary = {}
        for r in final_regions:
            Ov = region_stats2[r]['O']
            Dv = region_stats2[r]['D']
            Nv = region_stats2[r]['N']
            Ev = region_stats2[r]['Eint']
            imb = abs(Ov - Dv) / (Ov + Dv + 1e-6)
            print(f"  区域 {r}: N={Nv}, E={Ev}, O={Ov:.1f}, D={Dv:.1f}, imbalance={imb:.3f}")
            region_summary[int(r)] = {
                'nodes': int(Nv),
                'internal_edges': int(Ev),
                'O': float(Ov),
                'D': float(Dv),
                'imbalance': float(imb)
            }
        summary_file = f'region_summary_alpha_{ALPHA_VALUE}_od_{BETA_VALUE}.json'
        with open(summary_file, 'w') as f:
            json.dump(region_summary, f, indent=4)
        print(f"区域摘要已保存到: {summary_file}")

        visualize_partition(graph, constrained_membership, nodes_coords, alpha=ALPHA_VALUE)
        
        output_data = {graph.vs[i]['id']: membership for i, membership in enumerate(constrained_membership)}
        output_json_path = f'partition_results_alpha_{ALPHA_VALUE}_od_{BETA_VALUE}.json'
        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"详细的分区结果已保存到 {output_json_path}")
        
        # === 新增：生成边-区域映射和边邻接关系 ===
        generate_edge_mappings_and_adjacency(sumo_edges, output_data, ALPHA_VALUE)
        
    except FileNotFoundError:
        print(f"错误: 未找到文件 '{SUMO_NET_FILE}'。请确保文件与脚本在同一目录下。")
    except Exception as e:
        print(f"发生了一个错误: {e}")


