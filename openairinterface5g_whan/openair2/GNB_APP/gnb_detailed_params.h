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

/*! \file gnb_detailed_params.h
 * \brief Parameter definitions for CSI-RS/SRS/PMI detailed configuration
 * \author AI Assistant
 * \date 2024
 * \version 1.0
 * \company OAI
 * \email: contact@openairinterface.org
 */

 #ifndef __GNB_DETAILED_PARAMS_H__
 #define __GNB_DETAILED_PARAMS_H__
 
 #include "common/config/config_paramdesc.h"
 
 // CSI-RS 상세 설정 파라미터 정의
 #define CSIRSPARAMS_DESC { \
     {GNB_CONFIG_STRING_CSIRS_PERIODICITY, "CSI-RS 주기성 (슬롯 단위)", 0, {.uptr = NULL}, {.defuintval = 0}, TYPE_UINT32, 0, NULL, NULL}, \
     {GNB_CONFIG_STRING_CSIRS_FIRST_SYMBOL, "첫 번째 심볼 위치", 0, {.uptr = NULL}, {.defuintval = 0}, TYPE_UINT32, 0, NULL, NULL}, \
     {GNB_CONFIG_STRING_CSIRS_POWER_OFFSET, "파워 오프셋 (dB)", 0, {.uptr = NULL}, {.defuintval = 0}, TYPE_UINT32, 0, NULL, NULL}, \
     {GNB_CONFIG_STRING_CSIRS_DENSITY, "밀도 (1=one, 3=three)", 0, {.uptr = NULL}, {.defuintval = 1}, TYPE_UINT32, 0, NULL, NULL}, \
     {GNB_CONFIG_STRING_CSIRS_FREQ_START_RB, "시작 RB", 0, {.uptr = NULL}, {.defuintval = 0}, TYPE_UINT32, 0, NULL, NULL}, \
     {GNB_CONFIG_STRING_CSIRS_FREQ_NROF_RBS, "RB 수", 0, {.uptr = NULL}, {.defuintval = 106}, TYPE_UINT32, 0, NULL, NULL}, \
     {GNB_CONFIG_STRING_CSIRS_FREQ_ALLOCATION, "주파수 할당 [0]", 0, {.uptr = NULL}, {.defuintval = 0}, TYPE_UINT32, 0, NULL, NULL}, \
     {GNB_CONFIG_STRING_CSIRS_FREQ_ALLOCATION, "주파수 할당 [1]", 0, {.uptr = NULL}, {.defuintval = 16}, TYPE_UINT32, 0, NULL, NULL}, \
     {GNB_CONFIG_STRING_CSIRS_CDM_TYPE, "CDM 타입", 0, {.strptr = NULL}, {.defstrval = "fd_CDM2"}, TYPE_STRING, 0, NULL, NULL}, \
     {GNB_CONFIG_STRING_CSIRS_NROF_PORTS, "포트 수", 0, {.uptr = NULL}, {.defuintval = 2}, TYPE_UINT32, 0, NULL, NULL}, \
     {GNB_CONFIG_STRING_CSIRS_QCL_INFO, "QCL 정보", 0, {.uptr = NULL}, {.defuintval = 0}, TYPE_UINT32, 0, NULL, NULL}, \
 }
 
 // SRS 상세 설정 파라미터 정의
 #define SRSPARAMS_DESC { \
     {GNB_CONFIG_STRING_SRS_PERIODICITY, "SRS 주기성 (슬롯 단위)", 0, {.uptr = NULL}, {.defuintval = 10}, TYPE_UINT32, 0, NULL, NULL}, \
     {GNB_CONFIG_STRING_SRS_START_POSITION, "시작 위치", 0, {.uptr = NULL}, {.defuintval = 0}, TYPE_UINT32, 0, NULL, NULL}, \
     {GNB_CONFIG_STRING_SRS_NUM_SYMBOLS, "심볼 수", 0, {.uptr = NULL}, {.defuintval = 1}, TYPE_UINT32, 0, NULL, NULL}, \
     {GNB_CONFIG_STRING_SRS_REPETITION_FACTOR, "반복 인자", 0, {.uptr = NULL}, {.defuintval = 1}, TYPE_UINT32, 0, NULL, NULL}, \
     {GNB_CONFIG_STRING_SRS_COMB_OFFSET, "콤 오프셋", 0, {.uptr = NULL}, {.defuintval = 0}, TYPE_UINT32, 0, NULL, NULL}, \
     {GNB_CONFIG_STRING_SRS_CYCLIC_SHIFT, "순환 시프트", 0, {.uptr = NULL}, {.defuintval = 0}, TYPE_UINT32, 0, NULL, NULL}, \
     {GNB_CONFIG_STRING_SRS_FREQ_DOMAIN_POS, "주파수 도메인 위치", 0, {.uptr = NULL}, {.defuintval = 0}, TYPE_UINT32, 0, NULL, NULL}, \
     {GNB_CONFIG_STRING_SRS_FREQ_DOMAIN_SHIFT, "주파수 도메인 시프트", 0, {.uptr = NULL}, {.defuintval = 0}, TYPE_UINT32, 0, NULL, NULL}, \
     {GNB_CONFIG_STRING_SRS_FREQ_HOPPING_B_SRS, "주파수 호핑 b_SRS", 0, {.uptr = NULL}, {.defuintval = 0}, TYPE_UINT32, 0, NULL, NULL}, \
     {GNB_CONFIG_STRING_SRS_FREQ_HOPPING_B_HOP, "주파수 호핑 b_hop", 0, {.uptr = NULL}, {.defuintval = 0}, TYPE_UINT32, 0, NULL, NULL}, \
     {GNB_CONFIG_STRING_SRS_FREQ_HOPPING_C_SRS, "주파수 호핑 c_SRS", 0, {.uptr = NULL}, {.defuintval = 0}, TYPE_UINT32, 0, NULL, NULL}, \
     {GNB_CONFIG_STRING_SRS_GROUP_HOPPING, "그룹 호핑", 0, {.strptr = NULL}, {.defstrval = "neither"}, TYPE_STRING, 0, NULL, NULL}, \
     {GNB_CONFIG_STRING_SRS_ALPHA, "알파", 0, {.strptr = NULL}, {.defstrval = "alpha1"}, TYPE_STRING, 0, NULL, NULL}, \
     {GNB_CONFIG_STRING_SRS_P0, "P0 값 (dBm)", 0, {.iptr = NULL}, {.defintval = -84}, TYPE_INT32, 0, NULL, NULL}, \
     {GNB_CONFIG_STRING_SRS_NROF_PORTS, "SRS 포트 수", 0, {.uptr = NULL}, {.defuintval = 1}, TYPE_UINT32, 0, NULL, NULL}, \
 }
 
 // 코드북 상세 설정 파라미터 정의
 #define CODEBOOKPARAMS_DESC { \
     {GNB_CONFIG_STRING_CODEBOOK_TYPE, "코드북 타입", 0, {.strptr = NULL}, {.defstrval = "type1"}, TYPE_STRING, 0, NULL, NULL}, \
     {GNB_CONFIG_STRING_CODEBOOK_SUB_TYPE, "서브 타입", 0, {.strptr = NULL}, {.defstrval = "typeI_SinglePanel"}, TYPE_STRING, 0, NULL, NULL}, \
     {GNB_CONFIG_STRING_CODEBOOK_MODE, "모드", 0, {.uptr = NULL}, {.defuintval = 1}, TYPE_UINT32, 0, NULL, NULL}, \
     {GNB_CONFIG_STRING_CODEBOOK_PMI_RESTRICTION, "PMI 제한", 0, {.uptr = NULL}, {.defuintval = 0xff}, TYPE_UINT32, 0, NULL, NULL}, \
     {GNB_CONFIG_STRING_CODEBOOK_RI_RESTRICTION, "RI 제한", 0, {.uptr = NULL}, {.defuintval = 0x3}, TYPE_UINT32, 0, NULL, NULL}, \
     {GNB_CONFIG_STRING_CODEBOOK_N1_N2_CONFIG, "N1/N2 설정", 0, {.strptr = NULL}, {.defstrval = "two_one"}, TYPE_STRING, 0, NULL, NULL}, \
     {GNB_CONFIG_STRING_CODEBOOK_PHASE_ALPHABET, "Phase alphabet (n4=QPSK, n8=8PSK)", 0, {.strptr = NULL}, {.defstrval = "n4"}, TYPE_STRING, 0, NULL, NULL}, \
     {GNB_CONFIG_STRING_CODEBOOK_SUBBAND_AMP, "Subband amplitude reporting", 0, {.uptr = NULL}, {.defuintval = 0}, TYPE_UINT32, 0, NULL, NULL}, \
     {GNB_CONFIG_STRING_CODEBOOK_NUM_BEAMS, "Number of beams L (2,3,4)", 0, {.uptr = NULL}, {.defuintval = 2}, TYPE_UINT32, 0, NULL, NULL}, \
     {GNB_CONFIG_STRING_CODEBOOK_PORT_SEL_SIZE, "Port selection sampling size d (1,2,3,4)", 0, {.uptr = NULL}, {.defuintval = 2}, TYPE_UINT32, 0, NULL, NULL}, \
 }
 
 // CSI 측정 설정 파라미터 정의
 #define CSIPARAMS_DESC { \
     {GNB_CONFIG_STRING_CSI_CQI_TABLE, "CQI 테이블", 0, {.strptr = NULL}, {.defstrval = "table1"}, TYPE_STRING, 0, NULL, NULL}, \
     {GNB_CONFIG_STRING_CSI_RI_TABLE, "RI 테이블", 0, {.strptr = NULL}, {.defstrval = "table1"}, TYPE_STRING, 0, NULL, NULL}, \
     {GNB_CONFIG_STRING_CSI_PMI_TABLE, "PMI 테이블", 0, {.strptr = NULL}, {.defstrval = "table1"}, TYPE_STRING, 0, NULL, NULL}, \
     {GNB_CONFIG_STRING_CSI_REPORT_PERIODICITY, "보고 주기성", 0, {.uptr = NULL}, {.defuintval = 20}, TYPE_UINT32, 0, NULL, NULL}, \
     {GNB_CONFIG_STRING_CSI_REPORT_OFFSET, "보고 오프셋", 0, {.uptr = NULL}, {.defuintval = 0}, TYPE_UINT32, 0, NULL, NULL}, \
     {GNB_CONFIG_STRING_CSI_CQI_THRESHOLD, "CQI 임계값", 0, {.uptr = NULL}, {.defuintval = 0}, TYPE_UINT32, 0, NULL, NULL}, \
     {GNB_CONFIG_STRING_CSI_RI_THRESHOLD, "RI 임계값", 0, {.uptr = NULL}, {.defuintval = 0}, TYPE_UINT32, 0, NULL, NULL}, \
 }
 
 #endif 