#!/usr/bin/env python3
"""
디지털 트윈용 지도 파일 처리 및 시각화 도구

지원 형식:
- OSM (OpenStreetMap) .osm 파일
- GeoJSON .geojson 파일
- Shapefile .shp 파일
- Blender .blend 파일
- 3D 모델 (OBJ, PLY, FBX 등)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 옵션 라이브러리들 (필요시 설치)
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("[INFO] plotly 미설치 - 인터랙티브 3D 시각화 불가 (pip install plotly로 설치 가능)")

try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    print("[INFO] geopandas 미설치 - GeoJSON/Shapefile 처리 불가 (pip install geopandas로 설치 가능)")

try:
    import xml.etree.ElementTree as ET
    XML_AVAILABLE = True
except ImportError:
    XML_AVAILABLE = False

try:
    from sionna.rt import load_scene
    SIONNA_AVAILABLE = True
except ImportError:
    SIONNA_AVAILABLE = False
    print("[INFO] sionna 미설치 - Sionna 씬 로드 불가")


class MapVisualizer:
    """지도 파일 처리 및 시각화 클래스"""
    
    def __init__(self, map_file_path):
        """
        Args:
            map_file_path: 지도 파일 경로
        """
        self.map_file_path = Path(map_file_path)
        if not self.map_file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {map_file_path}")
        
        self.file_ext = self.map_file_path.suffix.lower()
        self.data = None
        self.scene = None
        
    def detect_file_type(self):
        """파일 형식 자동 감지"""
        ext_to_type = {
            '.osm': 'osm',
            '.geojson': 'geojson',
            '.json': 'geojson',  # GeoJSON도 .json 확장자 사용 가능
            '.shp': 'shapefile',
            '.blend': 'blender',
            '.xml': 'sionna_xml',  # Sionna/Mitsuba XML
            '.obj': 'obj',
            '.ply': 'ply',
            '.fbx': 'fbx',
        }
        return ext_to_type.get(self.file_ext, 'unknown')
    
    def load_osm(self):
        """OSM 파일 로드"""
        if not XML_AVAILABLE:
            raise ImportError("xml.etree.ElementTree 필요 (표준 라이브러리)")
        
        print(f"[OSM] 파일 로드 중: {self.map_file_path}")
        tree = ET.parse(self.map_file_path)
        root = tree.getroot()
        
        # OSM 데이터 파싱
        nodes = {}
        ways = []
        
        for elem in root:
            if elem.tag == 'node':
                node_id = elem.get('id')
                nodes[node_id] = {
                    'lat': float(elem.get('lat')),
                    'lon': float(elem.get('lon')),
                    'tags': {tag.get('k'): tag.get('v') for tag in elem.findall('tag')}
                }
            elif elem.tag == 'way':
                way_nodes = [nd.get('ref') for nd in elem.findall('nd')]
                way_tags = {tag.get('k'): tag.get('v') for tag in elem.findall('tag')}
                ways.append({
                    'nodes': way_nodes,
                    'tags': way_tags
                })
        
        self.data = {
            'type': 'osm',
            'nodes': nodes,
            'ways': ways
        }
        print(f"[OSM] 로드 완료: {len(nodes)}개 노드, {len(ways)}개 웨이")
        return self.data
    
    def load_geojson(self):
        """GeoJSON 파일 로드"""
        if not GEOPANDAS_AVAILABLE:
            raise ImportError("geopandas 필요 (pip install geopandas)")
        
        print(f"[GeoJSON] 파일 로드 중: {self.map_file_path}")
        gdf = gpd.read_file(self.map_file_path)
        self.data = {
            'type': 'geojson',
            'gdf': gdf,
            'bounds': gdf.total_bounds,
            'crs': gdf.crs
        }
        print(f"[GeoJSON] 로드 완료: {len(gdf)}개 피처")
        print(f"[GeoJSON] 좌표계: {gdf.crs}")
        print(f"[GeoJSON] 범위: {gdf.total_bounds}")
        return self.data
    
    def load_shapefile(self):
        """Shapefile 로드"""
        if not GEOPANDAS_AVAILABLE:
            raise ImportError("geopandas 필요 (pip install geopandas)")
        
        print(f"[Shapefile] 파일 로드 중: {self.map_file_path}")
        # .shp 파일이 있으면 자동으로 .shx, .dbf 등도 함께 로드됨
        gdf = gpd.read_file(self.map_file_path)
        self.data = {
            'type': 'shapefile',
            'gdf': gdf,
            'bounds': gdf.total_bounds,
            'crs': gdf.crs
        }
        print(f"[Shapefile] 로드 완료: {len(gdf)}개 피처")
        return self.data
    
    def load_sionna_xml(self):
        """Sionna/Mitsuba XML 씬 파일 로드"""
        if not SIONNA_AVAILABLE:
            raise ImportError("sionna 필요 (pip install sionna)")
        
        print(f"[Sionna XML] 씬 로드 중: {self.map_file_path}")
        # 상대 경로로 변환 (Sionna는 상대 경로 기반)
        scene_dir = self.map_file_path.parent
        xml_name = self.map_file_path.name
        
        # 작업 디렉토리 변경
        original_cwd = os.getcwd()
        try:
            os.chdir(scene_dir)
            self.scene = load_scene(xml_name, merge_shapes=False)
            self.data = {
                'type': 'sionna_xml',
                'scene': self.scene,
                'objects': list(self.scene.objects),
                'num_transmitters': len(self.scene.transmitters),
                'num_receivers': len(self.scene.receivers)
            }
            print(f"[Sionna XML] 로드 완료: {len(self.scene.objects)}개 객체")
        finally:
            os.chdir(original_cwd)
        
        return self.data
    
    def load(self):
        """파일 형식에 따라 자동 로드"""
        file_type = self.detect_file_type()
        print(f"[INFO] 파일 형식 감지: {file_type}")
        
        if file_type == 'osm':
            return self.load_osm()
        elif file_type == 'geojson':
            return self.load_geojson()
        elif file_type == 'shapefile':
            return self.load_shapefile()
        elif file_type == 'sionna_xml':
            return self.load_sionna_xml()
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {file_type}")
    
    def visualize_2d_matplotlib(self, save_path=None, show=True):
        """2D 시각화 (matplotlib)"""
        if self.data is None:
            self.load()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        if self.data['type'] == 'osm':
            # OSM 노드들을 점으로 표시
            lats = [node['lat'] for node in self.data['nodes'].values()]
            lons = [node['lon'] for node in self.data['nodes'].values()]
            ax.scatter(lons, lats, s=1, alpha=0.5, c='blue', label='OSM Nodes')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title('OpenStreetMap 데이터')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        elif self.data['type'] in ['geojson', 'shapefile']:
            # GeoDataFrame 플롯
            gdf = self.data['gdf']
            gdf.plot(ax=ax, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.set_xlabel('X (Longitude)')
            ax.set_ylabel('Y (Latitude)')
            ax.set_title(f'{self.data["type"].upper()} 데이터')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[시각화] 저장 완료: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def visualize_3d_plotly(self, save_html=None, show=True):
        """3D 인터랙티브 시각화 (plotly)"""
        if not PLOTLY_AVAILABLE:
            print("[경고] plotly 미설치 - 3D 시각화 불가")
            return
        
        if self.data is None:
            self.load()
        
        fig = go.Figure()
        
        if self.data['type'] == 'osm':
            # OSM 노드들을 3D로 표시
            lats = [node['lat'] for node in self.data['nodes'].values()]
            lons = [node['lon'] for node in self.data['nodes'].values()]
            # 고도는 0으로 설정 (OSM에는 고도 정보가 없을 수 있음)
            alts = [0] * len(lats)
            
            fig.add_trace(go.Scatter3d(
                x=lons, y=lats, z=alts,
                mode='markers',
                marker=dict(size=2, color='blue', opacity=0.6),
                name='OSM Nodes'
            ))
            
            fig.update_layout(
                title='OpenStreetMap 3D 시각화',
                scene=dict(
                    xaxis_title='Longitude',
                    yaxis_title='Latitude',
                    zaxis_title='Altitude'
                )
            )
        
        elif self.data['type'] in ['geojson', 'shapefile']:
            # GeoDataFrame을 3D로 표시 (고도 정보가 있으면 사용)
            gdf = self.data['gdf']
            
            # Z 좌표 추출 (고도 컬럼이 있으면 사용, 없으면 0)
            if 'z' in gdf.columns or 'Z' in gdf.columns:
                z_col = 'z' if 'z' in gdf.columns else 'Z'
            elif 'elevation' in gdf.columns:
                z_col = 'elevation'
            else:
                z_col = None
            
            # 각 지오메트리 타입에 따라 처리
            for idx, row in gdf.iterrows():
                geom = row.geometry
                if geom.type == 'Point':
                    x, y = geom.x, geom.y
                    z = row[z_col] if z_col else 0
                    fig.add_trace(go.Scatter3d(
                        x=[x], y=[y], z=[z],
                        mode='markers',
                        marker=dict(size=5),
                        name=f'Point {idx}'
                    ))
                elif geom.type in ['LineString', 'MultiLineString']:
                    # 라인 스트링 처리
                    coords = list(geom.coords) if geom.type == 'LineString' else \
                             [list(linestring.coords) for linestring in geom.geoms]
                    for coord_list in coords if isinstance(coords[0], list) else [coords]:
                        x = [c[0] for c in coord_list]
                        y = [c[1] for c in coord_list]
                        z = [c[2] if len(c) > 2 else 0 for c in coord_list]
                        fig.add_trace(go.Scatter3d(
                            x=x, y=y, z=z,
                            mode='lines',
                            line=dict(width=2),
                            name=f'Line {idx}'
                        ))
        
        elif self.data['type'] == 'sionna_xml':
            # Sionna 씬의 객체들을 3D로 표시
            scene = self.scene
            
            # Transmitter 위치
            if len(scene.transmitters) > 0:
                tx_positions = [list(tx.position.numpy()) for tx in scene.transmitters]
                tx_x = [pos[0] for pos in tx_positions]
                tx_y = [pos[1] for pos in tx_positions]
                tx_z = [pos[2] for pos in tx_positions]
                fig.add_trace(go.Scatter3d(
                    x=tx_x, y=tx_y, z=tx_z,
                    mode='markers',
                    marker=dict(size=10, color='red', symbol='diamond'),
                    name='Transmitters'
                ))
            
            # Receiver 위치
            if len(scene.receivers) > 0:
                rx_positions = [list(rx.position.numpy()) for rx in scene.receivers]
                rx_x = [pos[0] for pos in rx_positions]
                rx_y = [pos[1] for pos in rx_positions]
                rx_z = [pos[2] for pos in rx_positions]
                fig.add_trace(go.Scatter3d(
                    x=rx_x, y=rx_y, z=rx_z,
                    mode='markers',
                    marker=dict(size=8, color='green', symbol='circle'),
                    name='Receivers'
                ))
            
            fig.update_layout(
                title='Sionna 씬 3D 시각화',
                scene=dict(
                    xaxis_title='X (m)',
                    yaxis_title='Y (m)',
                    zaxis_title='Z (m)',
                    aspectmode='data'
                )
            )
        
        if save_html:
            fig.write_html(save_html)
            print(f"[시각화] HTML 저장 완료: {save_html}")
        
        if show:
            fig.show()
        
        return fig
    
    def visualize_sionna_preview(self):
        """Sionna 내장 preview 사용"""
        if self.data is None:
            self.load()
        
        if self.data['type'] != 'sionna_xml':
            raise ValueError("Sionna preview는 XML 씬 파일에만 사용 가능")
        
        print("[Sionna] 씬 미리보기 실행 중...")
        self.scene.preview()
    
    def get_summary(self):
        """데이터 요약 정보 반환"""
        if self.data is None:
            self.load()
        
        summary = {
            'file_path': str(self.map_file_path),
            'file_type': self.detect_file_type(),
            'data_type': self.data['type']
        }
        
        if self.data['type'] == 'osm':
            summary['num_nodes'] = len(self.data['nodes'])
            summary['num_ways'] = len(self.data['ways'])
        elif self.data['type'] in ['geojson', 'shapefile']:
            summary['num_features'] = len(self.data['gdf'])
            summary['bounds'] = self.data['bounds'].tolist()
            summary['crs'] = str(self.data['crs'])
        elif self.data['type'] == 'sionna_xml':
            summary['num_objects'] = len(self.data['objects'])
            summary['num_transmitters'] = self.data['num_transmitters']
            summary['num_receivers'] = self.data['num_receivers']
        
        return summary


def main():
    """메인 함수 - 명령줄에서 실행"""
    if len(sys.argv) < 2:
        print("사용법: python visualize_map_data.py <지도파일경로> [옵션]")
        print("\n옵션:")
        print("  --2d          : 2D matplotlib 시각화")
        print("  --3d          : 3D plotly 시각화")
        print("  --sionna      : Sionna preview (XML 파일만)")
        print("  --save-html   : HTML 파일로 저장 (3D만)")
        print("  --summary     : 데이터 요약 정보만 출력")
        return
    
    map_file = sys.argv[1]
    options = sys.argv[2:] if len(sys.argv) > 2 else []
    
    try:
        visualizer = MapVisualizer(map_file)
        
        if '--summary' in options:
            summary = visualizer.get_summary()
            print("\n=== 데이터 요약 ===")
            for key, value in summary.items():
                print(f"{key}: {value}")
            return
        
        # 기본적으로 모든 시각화 실행
        if '--2d' in options or len(options) == 0:
            visualizer.visualize_2d_matplotlib(show=True)
        
        if '--3d' in options or len(options) == 0:
            save_html = None
            if '--save-html' in options:
                save_html = str(Path(map_file).stem) + '_3d.html'
            visualizer.visualize_3d_plotly(save_html=save_html, show=True)
        
        if '--sionna' in options and visualizer.detect_file_type() == 'sionna_xml':
            visualizer.visualize_sionna_preview()
    
    except Exception as e:
        print(f"[오류] {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

