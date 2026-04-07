/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.1  (the "License"); you may not use this file
 * except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.openairinterface.org/?page_id=698
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *-------------------------------------------------------------------------------
 * For more information about the OpenAirInterface (OAI) Software Alliance:
 *      contact@openairinterface.org
 */

/*! \file nr_detailed_config.h
 * \brief Detailed configuration structures for CSI-RS, SRS, and Codebook settings
 * \author AI Assistant
 * \date 2024
 * \version 1.0
 * \company OAI
 * \email: contact@openairinterface.org
 */

 #ifndef __NR_DETAILED_CONFIG_H__
 #define __NR_DETAILED_CONFIG_H__
 
 #include <stdint.h>
 
 // CSI-RS 상세 설정 구조체
 typedef struct {
     int periodicity;           // 슬롯 단위 주기성 (4,5,8,10,16,20,40,80,160,320)
     int first_symbol;          // 첫 번째 심볼 위치
     int power_offset;          // 파워 오프셋 (dB)
     int density;               // 밀도 (1=one, 3=three)
     int freq_start_rb;         // 시작 RB
     int freq_nrof_rbs;         // RB 수
     int freq_allocation[2];    // 주파수 할당 (안테나 포트별)
     char cdm_type[20];         // CDM 타입 (noCDM, fd_CDM2, cd_CDM4, cd_CDM8)
     int nrof_ports;            // 포트 수 (1,2,4,8,12,16,24,32)
     int qcl_info;              // QCL 정보
 } csirs_detailed_config_t;
 
 // SRS 상세 설정 구조체
 typedef struct {
     int periodicity;           // 슬롯 단위 주기성
     int start_position;        // 시작 위치
     int num_symbols;           // 심볼 수 (1,2,4)
     int repetition_factor;     // 반복 인자 (1,2,4)
     int comb_offset;           // 콤 오프셋 (0,1,2,3)
     int cyclic_shift;          // 순환 시프트 (0-7)
     int freq_domain_position;  // 주파수 도메인 위치
     int freq_domain_shift;     // 주파수 도메인 시프트
     int freq_hopping_b_srs;    // 주파수 호핑 b_SRS
     int freq_hopping_b_hop;    // 주파수 호핑 b_hop
     int freq_hopping_c_srs;    // 주파수 호핑 c_SRS
     char group_hopping[20];    // 그룹 호핑 (neither, groupHopping, sequenceHopping)
     char alpha[20];            // 알파 (alpha1, alpha04, alpha06, alpha08)
     int p0;                    // P0 값 (dBm)
     int nrof_srs_ports;        // SRS 포트 수 (1,2,4)
 } srs_detailed_config_t;
 
 // 코드북 상세 설정 구조체
 typedef struct {
     char codebook_type[20];    // 코드북 타입 (type1, type2)
     char sub_type[30];         // 서브 타입 (typeI_SinglePanel, typeI_MultiPanel, typeII_PortSelection)
     int mode;                  // 모드 (1,2)
     int pmi_restriction;       // PMI 제한 (0xff = 제한 없음)
     int ri_restriction;        // RI 제한 (0x3 = 레이어 1,2 허용)
     char n1_n2_config[20];     // N1/N2 설정 (two_one, two_two, four_one, ...)
     // Type-II Port Selection fields
     char phase_alphabet[8];    // "n4" (QPSK) or "n8" (8PSK)
     int subband_amplitude;     // 0=off, 1=on
     int number_of_beams;       // L: 2, 3, or 4
     int port_selection_sampling_size; // d: 1, 2, 3, or 4
 } codebook_detailed_config_t;
 
 // CSI 측정 설정 구조체
 typedef struct {
     char cqi_table[20];        // CQI 테이블 (table1, table2, table3)
     char ri_table[20];         // RI 테이블 (table1, table2)
     char pmi_table[20];        // PMI 테이블 (table1, table2)
     int report_periodicity;    // 보고 주기성
     int report_offset;         // 보고 오프셋
     int cqi_threshold;         // CQI 임계값
     int ri_threshold;          // RI 임계값
 } csi_measurement_config_t;
 
 // 통합 설정 구조체
 typedef struct {
     csirs_detailed_config_t csirs;
     srs_detailed_config_t srs;
     codebook_detailed_config_t codebook;
     csi_measurement_config_t csi_measurement;
 } nr_detailed_config_t;
 
 #endif 