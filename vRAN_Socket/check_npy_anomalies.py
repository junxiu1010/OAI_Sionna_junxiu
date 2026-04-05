#!/usr/bin/env python3
"""
.npy 파일의 이상 데이터 체크 스크립트
- NaN 값
- Inf 값
- 극단값
- 상수값
"""

import numpy as np
import os
import sys
from pathlib import Path

def check_npy_file(filepath):
    """단일 .npy 파일 체크"""
    try:
        # 파일 로드
        data = np.load(filepath)
        
        # 기본 정보
        shape = data.shape
        dtype = data.dtype
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        
        # 복소수 타입 체크
        is_complex = np.iscomplexobj(data)
        
        # 이상 데이터 체크
        anomalies = []
        
        # 1. NaN 체크
        if is_complex:
            nan_count = np.sum(np.isnan(data.real)) + np.sum(np.isnan(data.imag))
        else:
            nan_count = np.sum(np.isnan(data))
        
        if nan_count > 0:
            anomalies.append(f"⚠️  NaN 값: {nan_count}개")
        
        # 2. Inf 체크
        if is_complex:
            inf_count = np.sum(np.isinf(data.real)) + np.sum(np.isinf(data.imag))
        else:
            inf_count = np.sum(np.isinf(data))
        
        if inf_count > 0:
            anomalies.append(f"⚠️  Inf 값: {inf_count}개")
        
        # 3. 상수값 체크 (모든 값이 동일)
        if data.size > 1:
            if is_complex:
                is_constant = (np.std(data.real) == 0) and (np.std(data.imag) == 0)
            else:
                is_constant = (np.std(data) == 0)
            
            if is_constant:
                anomalies.append(f"⚠️  상수값: 모든 값이 동일 (값={data.flat[0]})")
        
        # 4. 통계 정보
        if is_complex:
            real_min = np.nanmin(data.real) if not np.all(np.isnan(data.real)) else np.nan
            real_max = np.nanmax(data.real) if not np.all(np.isnan(data.real)) else np.nan
            imag_min = np.nanmin(data.imag) if not np.all(np.isnan(data.imag)) else np.nan
            imag_max = np.nanmax(data.imag) if not np.all(np.isnan(data.imag)) else np.nan
            magnitude = np.abs(data)
            mag_min = np.nanmin(magnitude) if not np.all(np.isnan(magnitude)) else np.nan
            mag_max = np.nanmax(magnitude) if not np.all(np.isnan(magnitude)) else np.nan
            mag_mean = np.nanmean(magnitude) if not np.all(np.isnan(magnitude)) else np.nan
        else:
            data_min = np.nanmin(data) if not np.all(np.isnan(data)) else np.nan
            data_max = np.nanmax(data) if not np.all(np.isnan(data)) else np.nan
            data_mean = np.nanmean(data) if not np.all(np.isnan(data)) else np.nan
        
        # 5. 극단값 체크 (magnitude 기준)
        if is_complex:
            if mag_max > 1e10:
                anomalies.append(f"⚠️  극단적으로 큰 값: magnitude max={mag_max:.2e}")
        else:
            if abs(data_max) > 1e10 or abs(data_min) > 1e10:
                anomalies.append(f"⚠️  극단적으로 큰 값: min={data_min:.2e}, max={data_max:.2e}")
        
        # 결과 출력
        filename = os.path.basename(filepath)
        status = "✅ 정상" if not anomalies else "❌ 이상 발견"
        
        print(f"\n{'='*70}")
        print(f"{status}: {filename}")
        print(f"{'='*70}")
        print(f"📁 경로: {filepath}")
        print(f"📊 크기: {size_mb:.2f} MB")
        print(f"📐 Shape: {shape}")
        print(f"🔢 Type: {dtype}")
        print(f"🔣 Complex: {is_complex}")
        
        if is_complex:
            print(f"📈 Real: [{real_min:.2e}, {real_max:.2e}]")
            print(f"📈 Imag: [{imag_min:.2e}, {imag_max:.2e}]")
            print(f"📈 Magnitude: min={mag_min:.2e}, max={mag_max:.2e}, mean={mag_mean:.2e}")
        else:
            print(f"📈 Range: [{data_min:.2e}, {data_max:.2e}]")
            print(f"📈 Mean: {data_mean:.2e}")
        
        if anomalies:
            print(f"\n🚨 이상 데이터:")
            for anomaly in anomalies:
                print(f"   {anomaly}")
        else:
            print(f"\n✅ 이상 없음")
        
        return len(anomalies) > 0
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ 오류: {os.path.basename(filepath)}")
        print(f"{'='*70}")
        print(f"📁 경로: {filepath}")
        print(f"⚠️  에러: {e}")
        return True

def main():
    # .npy 파일 검색
    search_paths = [
        "/home/dclcom45/vRAN_Socket/Minsoo_Channel_Data",
        "/home/dclcom45/vRAN_Socket/saved_rays_data",
        "/home/dclcom45/vRAN_Socket"
    ]
    
    npy_files = []
    for search_path in search_paths:
        if os.path.exists(search_path):
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    if file.endswith('.npy'):
                        filepath = os.path.join(root, file)
                        npy_files.append(filepath)
    
    # 중복 제거 및 정렬
    npy_files = sorted(set(npy_files))
    
    print("="*70)
    print("🔍 .npy 파일 이상 데이터 체크")
    print("="*70)
    print(f"총 {len(npy_files)}개 파일 발견\n")
    
    # 각 파일 체크
    problem_files = []
    for filepath in npy_files:
        has_problem = check_npy_file(filepath)
        if has_problem:
            problem_files.append(filepath)
    
    # 최종 요약
    print("\n" + "="*70)
    print("📊 최종 요약")
    print("="*70)
    print(f"✅ 정상 파일: {len(npy_files) - len(problem_files)}개")
    print(f"❌ 이상 파일: {len(problem_files)}개")
    
    if problem_files:
        print(f"\n🚨 이상이 발견된 파일 목록:")
        for filepath in problem_files:
            print(f"   - {os.path.basename(filepath)}")
    else:
        print(f"\n🎉 모든 파일이 정상입니다!")

if __name__ == "__main__":
    main()


