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

/***********************************************************************
*
* FILENAME    :  csi_rx.c
*
* MODULE      :
*
* DESCRIPTION :  function to receive the channel state information
*
************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "executables/nr-softmodem-common.h"
#include "nr_transport_proto_ue.h"
#include "PHY/phy_extern_nr_ue.h"
#include "PHY/NR_REFSIG/nr_refsig.h"
#include "common/utils/nr/nr_common.h"
#include "PHY/NR_UE_ESTIMATION/filt16a_32.h"

// Additional memory allocation, because of applying the filter and the memory offset to ensure memory alignment
#define FILTER_MARGIN 32

//#define NR_CSIRS_DEBUG
//#define NR_CSIIM_DEBUG

void nr_det_A_MF_2x2(int32_t *a_mf_00,
                     int32_t *a_mf_01,
                     int32_t *a_mf_10,
                     int32_t *a_mf_11,
                     int32_t *det_fin,
                     const unsigned short nb_rb) {

  simde__m128i ad_re_128, bc_re_128, det_re_128;

  simde__m128i *a_mf_00_128 = (simde__m128i *)a_mf_00;
  simde__m128i *a_mf_01_128 = (simde__m128i *)a_mf_01;
  simde__m128i *a_mf_10_128 = (simde__m128i *)a_mf_10;
  simde__m128i *a_mf_11_128 = (simde__m128i *)a_mf_11;
  simde__m128i *det_fin_128 = (simde__m128i *)det_fin;

  for (int rb = 0; rb<3*nb_rb; rb++) {

    //complex multiplication (I_a+jQ_a)(I_d+jQ_d) = (I_aI_d - Q_aQ_d) + j(Q_aI_d + I_aQ_d)
    //The imag part is often zero, we compute only the real part
    ad_re_128 = simde_mm_madd_epi16(oai_mm_conj(a_mf_00_128[0]), a_mf_11_128[0]); //Re: I_a0*I_d0 - Q_a1*Q_d1

    //complex multiplication (I_b+jQ_b)(I_c+jQ_c) = (I_bI_c - Q_bQ_c) + j(Q_bI_c + I_bQ_c)
    //The imag part is often zero, we compute only the real part
    bc_re_128 = simde_mm_madd_epi16(oai_mm_conj(a_mf_01_128[0]), a_mf_10_128[0]); //Re: I_b0*I_c0 - Q_b1*Q_c1

    det_re_128 = simde_mm_sub_epi32(ad_re_128, bc_re_128);

    //det in Q30 format
    det_fin_128[0] = simde_mm_abs_epi32(det_re_128);

    det_fin_128+=1;
    a_mf_00_128+=1;
    a_mf_01_128+=1;
    a_mf_10_128+=1;
    a_mf_11_128+=1;
  }
}

void nr_squared_matrix_element(int32_t *a,
                               int32_t *a_sq,
                               const unsigned short nb_rb) {
  simde__m128i *a_128 = (simde__m128i *)a;
  simde__m128i *a_sq_128 = (simde__m128i *)a_sq;
  for (int rb=0; rb<3*nb_rb; rb++) {
    a_sq_128[0] = simde_mm_madd_epi16(a_128[0], a_128[0]);
    a_sq_128+=1;
    a_128+=1;
  }
}

void nr_numer_2x2(int32_t *a_00_sq,
                  int32_t *a_01_sq,
                  int32_t *a_10_sq,
                  int32_t *a_11_sq,
                  int32_t *num_fin,
                  const unsigned short nb_rb) {
  simde__m128i *a_00_sq_128 = (simde__m128i *)a_00_sq;
  simde__m128i *a_01_sq_128 = (simde__m128i *)a_01_sq;
  simde__m128i *a_10_sq_128 = (simde__m128i *)a_10_sq;
  simde__m128i *a_11_sq_128 = (simde__m128i *)a_11_sq;
  simde__m128i *num_fin_128 = (simde__m128i *)num_fin;
  for (int rb=0; rb<3*nb_rb; rb++) {
    simde__m128i sq_a_plus_sq_d_128 = simde_mm_add_epi32(a_00_sq_128[0], a_11_sq_128[0]);
    simde__m128i sq_b_plus_sq_c_128 = simde_mm_add_epi32(a_01_sq_128[0], a_10_sq_128[0]);
    num_fin_128[0] = simde_mm_add_epi32(sq_a_plus_sq_d_128, sq_b_plus_sq_c_128);
    num_fin_128+=1;
    a_00_sq_128+=1;
    a_01_sq_128+=1;
    a_10_sq_128+=1;
    a_11_sq_128+=1;
  }
}

bool is_csi_rs_in_symbol(const fapi_nr_dl_config_csirs_pdu_rel15_t csirs_config_pdu, const int symbol) {

  bool ret = false;

  // 38.211-Table 7.4.1.5.3-1: CSI-RS locations within a slot
  switch(csirs_config_pdu.row){
    case 1:
    case 2:
    case 3:
    case 4:
    case 6:
    case 9:
      if(symbol == csirs_config_pdu.symb_l0) {
        ret = true;
      }
      break;
    case 5:
    case 7:
    case 8:
    case 10:
    case 11:
    case 12:
      if(symbol == csirs_config_pdu.symb_l0 || symbol == (csirs_config_pdu.symb_l0+1) ) {
        ret = true;
      }
      break;
    case 13:
    case 14:
    case 16:
    case 17:
      if(symbol == csirs_config_pdu.symb_l0 || symbol == (csirs_config_pdu.symb_l0+1) ||
          symbol == csirs_config_pdu.symb_l1 || symbol == (csirs_config_pdu.symb_l1+1)) {
        ret = true;
      }
      break;
    case 15:
    case 18:
      if(symbol == csirs_config_pdu.symb_l0 || symbol == (csirs_config_pdu.symb_l0+1) || symbol == (csirs_config_pdu.symb_l0+2) ) {
        ret = true;
      }
      break;
    default:
      AssertFatal(0==1, "Row %d is not valid for CSI Table 7.4.1.5.3-1\n", csirs_config_pdu.row);
  }

  return ret;
}

static int nr_get_csi_rs_signal(const PHY_VARS_NR_UE *ue,
                                const UE_nr_rxtx_proc_t *proc,
                                const fapi_nr_dl_config_csirs_pdu_rel15_t *csirs_config_pdu,
                                const nr_csi_info_t *nr_csi_info,
                                const csi_mapping_parms_t *csi_mapping,
                                const int CDM_group_size,
                                c16_t csi_rs_received_signal[][ue->frame_parms.samples_per_slot_wCP],
                                uint32_t *rsrp,
                                int *rsrp_dBm,
                                const c16_t rxdataF[][ue->frame_parms.samples_per_slot_wCP])
{
  const NR_DL_FRAME_PARMS *fp = &ue->frame_parms;
  uint16_t meas_count = 0;
  uint32_t rsrp_sum = 0;

  for (int ant_rx = 0; ant_rx < fp->nb_antennas_rx; ant_rx++) {

    for (int rb = csirs_config_pdu->start_rb; rb < (csirs_config_pdu->start_rb+csirs_config_pdu->nr_of_rbs); rb++) {

      // for freq density 0.5 checks if even or odd RB
      if(csirs_config_pdu->freq_density <= 1 && csirs_config_pdu->freq_density != (rb % 2)) {
        continue;
      }

      for (int cdm_id = 0; cdm_id < csi_mapping->size; cdm_id++) {
        for (int s = 0; s < CDM_group_size; s++)  {

          // loop over frequency resource elements within a group
          for (int kp = 0; kp <= csi_mapping->kprime; kp++) {

            uint16_t k = (fp->first_carrier_offset + (rb * NR_NB_SC_PER_RB) + csi_mapping->koverline[cdm_id] + kp) % fp->ofdm_symbol_size;

            // loop over time resource elements within a group
            for (int lp = 0; lp <= csi_mapping->lprime; lp++) {
              uint16_t symb = lp + csi_mapping->loverline[cdm_id];
              uint64_t symbol_offset = symb * fp->ofdm_symbol_size;
              const c16_t *rx_signal = &rxdataF[ant_rx][symbol_offset];
              c16_t *rx_csi_rs_signal = &csi_rs_received_signal[ant_rx][symbol_offset];
              rx_csi_rs_signal[k].r = rx_signal[k].r;
              rx_csi_rs_signal[k].i = rx_signal[k].i;

              rsrp_sum += (((int32_t)(rx_csi_rs_signal[k].r)*rx_csi_rs_signal[k].r) +
                           ((int32_t)(rx_csi_rs_signal[k].i)*rx_csi_rs_signal[k].i));

              meas_count++;

#ifdef NR_CSIRS_DEBUG
              int dataF_offset = proc->nr_slot_rx * fp->samples_per_slot_wCP;
              uint16_t port_tx = s + csi_mapping->j[cdm_id] * CDM_group_size;
              c16_t *tx_csi_rs_signal = &nr_csi_info->csi_rs_generated_signal[port_tx][symbol_offset + dataF_offset];
              LOG_I(NR_PHY,
                    "l,k (%2d,%4d) |\tport_tx %d (%4d,%4d)\tant_rx %d (%4d,%4d)\n",
                    symb,
                    k,
                    port_tx+3000,
                    tx_csi_rs_signal[k].r,
                    tx_csi_rs_signal[k].i,
                    ant_rx,
                    rx_csi_rs_signal[k].r,
                    rx_csi_rs_signal[k].i);
#endif
            }
          }
        }
      }
    }
  }


  *rsrp = rsrp_sum/meas_count;
  *rsrp_dBm = dB_fixed(*rsrp) + 30 - SQ15_SQUARED_NORM_FACTOR_DB
      - ((int)openair0_cfg[0].rx_gain[0] - (int)openair0_cfg[0].rx_gain_offset[0]) - dB_fixed(ue->frame_parms.ofdm_symbol_size);

#ifdef NR_CSIRS_DEBUG
  LOG_I(NR_PHY, "RSRP = %i (%i dBm)\n", *rsrp, *rsrp_dBm);
#endif

  return 0;
}

uint32_t calc_power_csirs(const uint16_t *x, const fapi_nr_dl_config_csirs_pdu_rel15_t *csirs_config_pdu)
{
  uint64_t sum_x = 0;
  uint64_t sum_x2 = 0;
  uint16_t size = 0;
  for (int rb = 0; rb < csirs_config_pdu->nr_of_rbs; rb++) {
    if (csirs_config_pdu->freq_density <= 1 && csirs_config_pdu->freq_density != ((rb + csirs_config_pdu->start_rb) % 2)) {
      continue;
    }
    sum_x = sum_x + x[rb];
    sum_x2 = sum_x2 + x[rb] * x[rb];
    size++;
  }
  return sum_x2 / size - (sum_x / size) * (sum_x / size);
}

static int nr_csi_rs_channel_estimation(
    const NR_DL_FRAME_PARMS *fp,
    const UE_nr_rxtx_proc_t *proc,
    const fapi_nr_dl_config_csirs_pdu_rel15_t *csirs_config_pdu,
    const nr_csi_info_t *nr_csi_info,
    const c16_t **csi_rs_generated_signal,
    const c16_t csi_rs_received_signal[][fp->samples_per_slot_wCP],
    const csi_mapping_parms_t *csi_mapping,
    const int CDM_group_size,
    uint8_t mem_offset,
    c16_t csi_rs_ls_estimated_channel[][csi_mapping->ports][fp->ofdm_symbol_size],
    c16_t csi_rs_estimated_channel_freq[][csi_mapping->ports][fp->ofdm_symbol_size + FILTER_MARGIN],
    int16_t *log2_re,
    int16_t *log2_maxh,
    uint32_t *noise_power)
{
  const int dataF_offset = proc->nr_slot_rx * fp->samples_per_slot_wCP;
  *noise_power = 0;
  int maxh = 0;
  int count = 0;

  for (int ant_rx = 0; ant_rx < fp->nb_antennas_rx; ant_rx++) {

    /// LS channel estimation

    for(uint16_t port_tx = 0; port_tx < csi_mapping->ports; port_tx++) {
      memset(csi_rs_ls_estimated_channel[ant_rx][port_tx], 0, fp->ofdm_symbol_size * sizeof(c16_t));
    }

    for (int rb = csirs_config_pdu->start_rb; rb < (csirs_config_pdu->start_rb+csirs_config_pdu->nr_of_rbs); rb++) {

      // for freq density 0.5 checks if even or odd RB
      if(csirs_config_pdu->freq_density <= 1 && csirs_config_pdu->freq_density != (rb % 2)) {
        continue;
      }

      for (int cdm_id = 0; cdm_id < csi_mapping->size; cdm_id++) {
        for (int s = 0; s < CDM_group_size; s++)  {

          uint16_t port_tx = s + csi_mapping->j[cdm_id] * CDM_group_size;

          // loop over frequency resource elements within a group
          for (int kp = 0; kp <= csi_mapping->kprime; kp++) {

            uint16_t kinit = (fp->first_carrier_offset + rb*NR_NB_SC_PER_RB) % fp->ofdm_symbol_size;
            uint16_t k = kinit + csi_mapping->koverline[cdm_id] + kp;

            // loop over time resource elements within a group
            for (int lp = 0; lp <= csi_mapping->lprime; lp++) {
              uint16_t symb = lp + csi_mapping->loverline[cdm_id];
              uint64_t symbol_offset = symb * fp->ofdm_symbol_size;
              const c16_t *tx_csi_rs_signal = &csi_rs_generated_signal[port_tx][symbol_offset+dataF_offset];
              const c16_t *rx_csi_rs_signal = &csi_rs_received_signal[ant_rx][symbol_offset];
              c16_t tmp = c16MulConjShift(tx_csi_rs_signal[k], rx_csi_rs_signal[k], nr_csi_info->csi_rs_generated_signal_bits);
              // This is not just the LS estimation for each (k,l), but also the sum of the different contributions
              // for the sake of optimizing the memory used.
              csi_rs_ls_estimated_channel[ant_rx][port_tx][kinit].r += tmp.r;
              csi_rs_ls_estimated_channel[ant_rx][port_tx][kinit].i += tmp.i;
            }
          }
        }
      }
    }

#ifdef NR_CSIRS_DEBUG
    for(int symb = 0; symb < NR_SYMBOLS_PER_SLOT; symb++) {
      if(!is_csi_rs_in_symbol(*csirs_config_pdu,symb)) {
        continue;
      }
      for(int k = 0; k < fp->ofdm_symbol_size; k++) {
        LOG_I(NR_PHY, "l,k (%2d,%4d) | ", symb, k);
        for(uint16_t port_tx = 0; port_tx < csi_mapping->ports; port_tx++) {
          uint64_t symbol_offset = symb * fp->ofdm_symbol_size;
          c16_t *tx_csi_rs_signal = (c16_t*)&csi_rs_generated_signal[port_tx][symbol_offset+dataF_offset];
          c16_t *rx_csi_rs_signal = (c16_t*)&csi_rs_received_signal[ant_rx][symbol_offset];
          c16_t *csi_rs_ls_estimated_channel16 = csi_rs_ls_estimated_channel[ant_rx][port_tx];
          printf("port_tx %d --> ant_rx %d, tx (%4d,%4d), rx (%4d,%4d), ls (%4d,%4d) | ",
                 port_tx+3000, ant_rx,
                 tx_csi_rs_signal[k].r, tx_csi_rs_signal[k].i,
                 rx_csi_rs_signal[k].r, rx_csi_rs_signal[k].i,
                 csi_rs_ls_estimated_channel16[k].r, csi_rs_ls_estimated_channel16[k].i);
        }
        printf("\n");
      }
    }
#endif

    /// Channel interpolation

    for(uint16_t port_tx = 0; port_tx < csi_mapping->ports; port_tx++) {
      memset(csi_rs_estimated_channel_freq[ant_rx][port_tx], 0, (fp->ofdm_symbol_size + FILTER_MARGIN) * sizeof(c16_t));
    }

    for (int rb = csirs_config_pdu->start_rb; rb < (csirs_config_pdu->start_rb+csirs_config_pdu->nr_of_rbs); rb++) {

      // for freq density 0.5 checks if even or odd RB
      if(csirs_config_pdu->freq_density <= 1 && csirs_config_pdu->freq_density != (rb % 2)) {
        continue;
      }

      count++;

      uint16_t k = (fp->first_carrier_offset + rb * NR_NB_SC_PER_RB) % fp->ofdm_symbol_size;
      uint16_t k_offset = k + mem_offset;
      for(uint16_t port_tx = 0; port_tx < csi_mapping->ports; port_tx++) {
        c16_t csi_rs_ls_estimated_channel16 = csi_rs_ls_estimated_channel[ant_rx][port_tx][k];
        c16_t *csi_rs_estimated_channel16 = &csi_rs_estimated_channel_freq[ant_rx][port_tx][k_offset];
        if( (k == 0) || (k == fp->first_carrier_offset) ) { // Start of OFDM symbol case or first occupied subcarrier case
          multadd_real_vector_complex_scalar(filt24_start, csi_rs_ls_estimated_channel16, csi_rs_estimated_channel16, 24);
        } else if(((k + NR_NB_SC_PER_RB) >= fp->ofdm_symbol_size) ||
                   (rb == (csirs_config_pdu->start_rb+csirs_config_pdu->nr_of_rbs-1))) { // End of OFDM symbol case or Last occupied subcarrier case
          multadd_real_vector_complex_scalar(filt24_end, csi_rs_ls_estimated_channel16, csi_rs_estimated_channel16 - 12, 24);
        } else { // Middle case
          multadd_real_vector_complex_scalar(filt24_middle, csi_rs_ls_estimated_channel16, csi_rs_estimated_channel16 - 12, 24);
        }
      }
    }

    /// Power noise estimation
    AssertFatal(csirs_config_pdu->nr_of_rbs > 0, " nr_of_rbs needs to be greater than 0\n");
    uint16_t noise_real[fp->nb_antennas_rx][csi_mapping->ports][csirs_config_pdu->nr_of_rbs];
    uint16_t noise_imag[fp->nb_antennas_rx][csi_mapping->ports][csirs_config_pdu->nr_of_rbs];
    for (int rb = csirs_config_pdu->start_rb; rb < (csirs_config_pdu->start_rb+csirs_config_pdu->nr_of_rbs); rb++) {
      if (csirs_config_pdu->freq_density <= 1 && csirs_config_pdu->freq_density != (rb % 2)) {
        continue;
      }
      uint16_t k = (fp->first_carrier_offset + rb*NR_NB_SC_PER_RB) % fp->ofdm_symbol_size;
      uint16_t k_offset = k + mem_offset;
      for(uint16_t port_tx = 0; port_tx < csi_mapping->ports; port_tx++) {
        c16_t *csi_rs_ls_estimated_channel16 = &csi_rs_ls_estimated_channel[ant_rx][port_tx][k];
        c16_t *csi_rs_estimated_channel16 = &csi_rs_estimated_channel_freq[ant_rx][port_tx][k_offset];
        noise_real[ant_rx][port_tx][rb-csirs_config_pdu->start_rb] = abs(csi_rs_ls_estimated_channel16->r-csi_rs_estimated_channel16->r);
        noise_imag[ant_rx][port_tx][rb-csirs_config_pdu->start_rb] = abs(csi_rs_ls_estimated_channel16->i-csi_rs_estimated_channel16->i);
        maxh = cmax3(maxh, abs(csi_rs_estimated_channel16->r), abs(csi_rs_estimated_channel16->i));
      }
    }
    for(uint16_t port_tx = 0; port_tx < csi_mapping->ports; port_tx++) {
      *noise_power += (calc_power_csirs(noise_real[ant_rx][port_tx], csirs_config_pdu) + calc_power_csirs(noise_imag[ant_rx][port_tx],csirs_config_pdu));
    }

#ifdef NR_CSIRS_DEBUG
    for(int k = 0; k < fp->ofdm_symbol_size; k++) {
      int rb = k >= fp->first_carrier_offset ?
               (k - fp->first_carrier_offset)/NR_NB_SC_PER_RB :
               (k + fp->ofdm_symbol_size - fp->first_carrier_offset)/NR_NB_SC_PER_RB;
      LOG_I(NR_PHY, "(k = %4d) |\t", k);
      for(uint16_t port_tx = 0; port_tx < csi_mapping->ports; port_tx++) {
        c16_t *csi_rs_ls_estimated_channel16 = &csi_rs_ls_estimated_channel[ant_rx][port_tx][0];
        c16_t *csi_rs_estimated_channel16 = &csi_rs_estimated_channel_freq[ant_rx][port_tx][mem_offset];
        printf("Channel port_tx %d --> ant_rx %d : ls (%4d,%4d), int (%4d,%4d), noise (%4d,%4d) | ",
               port_tx+3000, ant_rx,
               csi_rs_ls_estimated_channel16[k].r, csi_rs_ls_estimated_channel16[k].i,
               csi_rs_estimated_channel16[k].r, csi_rs_estimated_channel16[k].i,
               rb >= csirs_config_pdu->start_rb+csirs_config_pdu->nr_of_rbs ? 0 : noise_real[ant_rx][port_tx][rb-csirs_config_pdu->start_rb],
               rb >= csirs_config_pdu->start_rb+csirs_config_pdu->nr_of_rbs ? 0 : noise_imag[ant_rx][port_tx][rb-csirs_config_pdu->start_rb]);
      }
      printf("\n");
    }
#endif

  }

  *noise_power /= (fp->nb_antennas_rx * csi_mapping->ports);
  *log2_maxh = log2_approx(maxh - 1);
  *log2_re = log2_approx(count - 1);

#ifdef NR_CSIRS_DEBUG
  LOG_I(NR_PHY, "Noise power estimation based on CSI-RS: %i\n", *noise_power);
#endif
  return 0;
}

int nr_csi_rs_ri_estimation(const PHY_VARS_NR_UE *ue,
                            const fapi_nr_dl_config_csirs_pdu_rel15_t *csirs_config_pdu,
                            const nr_csi_info_t *nr_csi_info,
                            const uint8_t N_ports,
                            uint8_t mem_offset,
                            c16_t csi_rs_estimated_channel_freq[][N_ports][ue->frame_parms.ofdm_symbol_size + FILTER_MARGIN],
                            const int16_t log2_maxh,
                            uint8_t *rank_indicator)
{
  const NR_DL_FRAME_PARMS *frame_parms = &ue->frame_parms;
  const int16_t cond_dB_threshold = 5;
  int count = 0;
  *rank_indicator = 0;

  if (ue->frame_parms.nb_antennas_rx == 1 || N_ports == 1) {
    return 0;
  } else if (N_ports > 2) {
    // For N_ports > 2: estimate RI based on total received power across ports.
    // Compute per-port power and check if rank-2 is beneficial using
    // a simplified condition-number proxy on the strongest 2 ports.
    int64_t port_power[N_ports];
    memset(port_power, 0, sizeof(port_power));
    for (int rb = csirs_config_pdu->start_rb; rb < (csirs_config_pdu->start_rb + csirs_config_pdu->nr_of_rbs); rb++) {
      if (csirs_config_pdu->freq_density <= 1 && csirs_config_pdu->freq_density != (rb % 2))
        continue;
      uint16_t k = (frame_parms->first_carrier_offset + rb * NR_NB_SC_PER_RB) % frame_parms->ofdm_symbol_size;
      uint16_t k_off = k + mem_offset;
      for (int p = 0; p < N_ports; p++) {
        for (int ar = 0; ar < frame_parms->nb_antennas_rx; ar++) {
          const c16_t h = csi_rs_estimated_channel_freq[ar][p][k_off];
          port_power[p] += (int64_t)h.r * h.r + (int64_t)h.i * h.i;
        }
      }
    }
    // Find two strongest ports
    int p0 = 0, p1 = 1;
    if (port_power[p1] > port_power[p0]) { int t = p0; p0 = p1; p1 = t; }
    for (int p = 2; p < N_ports; p++) {
      if (port_power[p] > port_power[p0]) { p1 = p0; p0 = p; }
      else if (port_power[p] > port_power[p1]) { p1 = p; }
    }
    // Cross-correlation between strongest two ports
    int64_t cross_re = 0, cross_im = 0;
    int rb_cnt = 0;
    for (int rb = csirs_config_pdu->start_rb; rb < (csirs_config_pdu->start_rb + csirs_config_pdu->nr_of_rbs); rb++) {
      if (csirs_config_pdu->freq_density <= 1 && csirs_config_pdu->freq_density != (rb % 2))
        continue;
      rb_cnt++;
      uint16_t k = (frame_parms->first_carrier_offset + rb * NR_NB_SC_PER_RB) % frame_parms->ofdm_symbol_size;
      uint16_t k_off = k + mem_offset;
      for (int ar = 0; ar < frame_parms->nb_antennas_rx; ar++) {
        const c16_t h0 = csi_rs_estimated_channel_freq[ar][p0][k_off];
        const c16_t h1 = csi_rs_estimated_channel_freq[ar][p1][k_off];
        cross_re += (int64_t)h0.r * h1.r + (int64_t)h0.i * h1.i;
        cross_im += (int64_t)h0.i * h1.r - (int64_t)h0.r * h1.i;
      }
    }
    int64_t cross_power = cross_re / (rb_cnt > 0 ? rb_cnt : 1);
    cross_power = cross_power * cross_re / (rb_cnt > 0 ? rb_cnt : 1)
                + (cross_im / (rb_cnt > 0 ? rb_cnt : 1)) * (cross_im / (rb_cnt > 0 ? rb_cnt : 1));
    int64_t auto_power = (port_power[p0] / (rb_cnt > 0 ? rb_cnt : 1))
                       * (port_power[p1] / (rb_cnt > 0 ? rb_cnt : 1));
    // If cross-correlation is low relative to auto power, rank 2 is feasible
    if (auto_power > 0 && (cross_power * 4 < auto_power))
      *rank_indicator = 1;
    else
      *rank_indicator = 0;
    LOG_D(NR_PHY, "RI estimation (N_ports=%d): cross_power=%lld, auto_power=%lld, RI=%d\n",
          N_ports, (long long)cross_power, (long long)auto_power, *rank_indicator + 1);
    return 0;
  } else if( !(ue->frame_parms.nb_antennas_rx == 2 && N_ports == 2) ) {
    LOG_W(NR_PHY, "Rank indicator computation is not implemented for %i x %i system\n",
          ue->frame_parms.nb_antennas_rx, N_ports);
    return -1;
  }

  /* Example 2x2: Hh x H =
  *            | conjch00 conjch10 | x | ch00 ch01 | = | conjch00*ch00+conjch10*ch10 conjch00*ch01+conjch10*ch11 |
  *            | conjch01 conjch11 |   | ch10 ch11 |   | conjch01*ch00+conjch11*ch10 conjch01*ch01+conjch11*ch11 |
  */

  c16_t csi_rs_estimated_conjch_ch[frame_parms->nb_antennas_rx][N_ports][frame_parms->nb_antennas_rx][N_ports]
                                  [frame_parms->ofdm_symbol_size + FILTER_MARGIN] __attribute__((aligned(32)));
  int32_t csi_rs_estimated_A_MF[N_ports][N_ports][frame_parms->ofdm_symbol_size + FILTER_MARGIN] __attribute__((aligned(32)));
  int32_t csi_rs_estimated_A_MF_sq[N_ports][N_ports][frame_parms->ofdm_symbol_size + FILTER_MARGIN] __attribute__((aligned(32)));
  int32_t csi_rs_estimated_determ_fin[frame_parms->ofdm_symbol_size + FILTER_MARGIN] __attribute__((aligned(32)));
  int32_t csi_rs_estimated_numer_fin[frame_parms->ofdm_symbol_size + FILTER_MARGIN] __attribute__((aligned(32)));
  const uint8_t sum_shift = 1; // log2(2x2) = 2, which is a shift of 1 bit
  
  for (int rb = csirs_config_pdu->start_rb; rb < (csirs_config_pdu->start_rb+csirs_config_pdu->nr_of_rbs); rb++) {

    if (csirs_config_pdu->freq_density <= 1 && csirs_config_pdu->freq_density != (rb % 2)) {
      continue;
    }
    uint16_t k = (frame_parms->first_carrier_offset + rb*NR_NB_SC_PER_RB) % frame_parms->ofdm_symbol_size;
    uint16_t k_offset = k + mem_offset;

    for (int ant_rx_conjch = 0; ant_rx_conjch < frame_parms->nb_antennas_rx; ant_rx_conjch++) {
      for(uint16_t port_tx_conjch = 0; port_tx_conjch < N_ports; port_tx_conjch++) {
        for (int ant_rx_ch = 0; ant_rx_ch < frame_parms->nb_antennas_rx; ant_rx_ch++) {
          for(uint16_t port_tx_ch = 0; port_tx_ch < N_ports; port_tx_ch++) {

            // conjch x ch computation
            nr_conjch0_mult_ch1(&csi_rs_estimated_channel_freq[ant_rx_conjch][port_tx_conjch][k_offset],
                                &csi_rs_estimated_channel_freq[ant_rx_ch][port_tx_ch][k_offset],
                                &csi_rs_estimated_conjch_ch[ant_rx_conjch][port_tx_conjch][ant_rx_ch][port_tx_ch][k_offset],
                                1,
                                log2_maxh);

            // construct Hh x H elements
            if(ant_rx_conjch == ant_rx_ch) {
              nr_a_sum_b(
                  (c16_t *)&csi_rs_estimated_A_MF[port_tx_conjch][port_tx_ch][k_offset], (c16_t *)&csi_rs_estimated_conjch_ch[ant_rx_conjch][port_tx_conjch][ant_rx_ch][port_tx_ch][k_offset], 1);
            }
          }
        }
      }
    }

    // compute the determinant of A_MF (denominator)
    nr_det_A_MF_2x2(&csi_rs_estimated_A_MF[0][0][k_offset],
                    &csi_rs_estimated_A_MF[0][1][k_offset],
                    &csi_rs_estimated_A_MF[1][0][k_offset],
                    &csi_rs_estimated_A_MF[1][1][k_offset],
                    &csi_rs_estimated_determ_fin[k_offset],
                    1);

    // compute the square of A_MF (numerator)
    nr_squared_matrix_element(&csi_rs_estimated_A_MF[0][0][k_offset], &csi_rs_estimated_A_MF_sq[0][0][k_offset], 1);
    nr_squared_matrix_element(&csi_rs_estimated_A_MF[0][1][k_offset], &csi_rs_estimated_A_MF_sq[0][1][k_offset], 1);
    nr_squared_matrix_element(&csi_rs_estimated_A_MF[1][0][k_offset], &csi_rs_estimated_A_MF_sq[1][0][k_offset], 1);
    nr_squared_matrix_element(&csi_rs_estimated_A_MF[1][1][k_offset], &csi_rs_estimated_A_MF_sq[1][1][k_offset], 1);
    nr_numer_2x2(&csi_rs_estimated_A_MF_sq[0][0][k_offset],
                 &csi_rs_estimated_A_MF_sq[0][1][k_offset],
                 &csi_rs_estimated_A_MF_sq[1][0][k_offset],
                 &csi_rs_estimated_A_MF_sq[1][1][k_offset],
                 &csi_rs_estimated_numer_fin[k_offset],
                 1);

#ifdef NR_CSIRS_DEBUG
    for(uint16_t port_tx_conjch = 0; port_tx_conjch < N_ports; port_tx_conjch++) {
      for(uint16_t port_tx_ch = 0; port_tx_ch < N_ports; port_tx_ch++) {
        c16_t *csi_rs_estimated_A_MF_k = (c16_t *) &csi_rs_estimated_A_MF[port_tx_conjch][port_tx_ch][k_offset];
        LOG_I(NR_PHY, "(%i) csi_rs_estimated_A_MF[%i][%i] = (%i, %i)\n",
              k, port_tx_conjch, port_tx_ch, csi_rs_estimated_A_MF_k->r, csi_rs_estimated_A_MF_k->i);
        c16_t *csi_rs_estimated_A_MF_sq_k = (c16_t *) &csi_rs_estimated_A_MF_sq[port_tx_conjch][port_tx_ch][k_offset];
        LOG_I(NR_PHY, "(%i) csi_rs_estimated_A_MF_sq[%i][%i] = (%i, %i)\n",
              k, port_tx_conjch, port_tx_ch, csi_rs_estimated_A_MF_sq_k->r, csi_rs_estimated_A_MF_sq_k->i);
      }
    }
    LOG_I(NR_PHY, "(%i) csi_rs_estimated_determ_fin = %i\n", k, csi_rs_estimated_determ_fin[k_offset]);
    LOG_I(NR_PHY, "(%i) csi_rs_estimated_numer_fin = %i\n", k, csi_rs_estimated_numer_fin[k_offset]>>sum_shift);
#endif

    // compute the conditional number
    for (int sc_idx=0; sc_idx < NR_NB_SC_PER_RB; sc_idx++) {
      int8_t csi_rs_estimated_denum_db = dB_fixed(csi_rs_estimated_determ_fin[k_offset + sc_idx]);
      int8_t csi_rs_estimated_numer_db = dB_fixed(csi_rs_estimated_numer_fin[k_offset + sc_idx]>>sum_shift);
      int8_t cond_db = csi_rs_estimated_numer_db - csi_rs_estimated_denum_db;

#ifdef NR_CSIRS_DEBUG
      LOG_I(NR_PHY, "csi_rs_estimated_denum_db = %i\n", csi_rs_estimated_denum_db);
      LOG_I(NR_PHY, "csi_rs_estimated_numer_db = %i\n", csi_rs_estimated_numer_db);
      LOG_I(NR_PHY, "cond_db = %i\n", cond_db);
#endif

      if (cond_db < cond_dB_threshold) {
        count++;
      } else {
        count--;
      }
    }
  }

  // conditional number is lower than cond_dB_threshold in half on more REs
  if (count > 0) {
    *rank_indicator = 1;
  }

#ifdef NR_CSIRS_DEBUG
  LOG_I(NR_PHY, "count = %i\n", count);
  LOG_I(NR_PHY, "rank = %i\n", (*rank_indicator)+1);
#endif

  return 0;
}

int nr_csi_rs_pmi_estimation(const PHY_VARS_NR_UE *ue,
                             const fapi_nr_dl_config_csirs_pdu_rel15_t *csirs_config_pdu,
                             const nr_csi_info_t *nr_csi_info,
                             const uint8_t N_ports,
                             uint8_t mem_offset,
                             const c16_t csi_rs_estimated_channel_freq[][N_ports][ue->frame_parms.ofdm_symbol_size + FILTER_MARGIN],
                             const uint32_t interference_plus_noise_power,
                             const uint8_t rank_indicator,
                             const int16_t log2_re,
                             uint8_t *i1,
                             uint8_t *i2,
                             uint32_t *precoded_sinr_dB)
{
  LOG_I(NR_PHY, "PMI: Function entered with N_ports=%d, interference_plus_noise_power=%u, rank_indicator=%d\n", 
        N_ports, interference_plus_noise_power, rank_indicator);
  const NR_DL_FRAME_PARMS *frame_parms = &ue->frame_parms;

  // i1 is a three-element vector in the form of [i11 i12 i13], when CodebookType is specified as 'Type1SinglePanel'.
  // Note that i13 is not applicable when the number of transmission layers is one of {1, 5, 6, 7, 8}.
  // i2, for 'Type1SinglePanel' codebook type, it is a scalar when PMIMode is specified as 'wideband', and when PMIMode
  // is specified as 'subband' or when PRGSize, the length of the i2 vector equals to the number of subbands or PRGs.
  // Note that when the number of CSI-RS ports is 2, the applicable codebook type is 'Type1SinglePanel'. In this case,
  // the precoding matrix is obtained by a single index (i2 field here) based on TS 38.214 Table 5.2.2.2.1-1.
  // The first column is applicable if the UE is reporting a Rank = 1, whereas the second column is applicable if the
  // UE is reporting a Rank = 2.

  if(N_ports == 1) {
    // 단일 포트: PMI는 의미 없지만, SINR은 실제 계산해야 CQI가 정확함
    // (기존 코드는 SINR=46 dB hardcoding → SNR 제어가 CQI에 반영 안 됨)
    i2[0] = 0;

    uint32_t effective_noise_power = interference_plus_noise_power;
    if (effective_noise_power == 0) {
      effective_noise_power = 1;
      LOG_I(NR_PHY, "PMI: Single port - noise power is 0, using minimum 1\n");
    }

    // 채널 추정치로부터 signal power 계산: Σ_rb Σ_ant |H(rb)|²
    int64_t signal_power = 0;
    int rb_count = 0;
    for (int rb = csirs_config_pdu->start_rb;
         rb < (csirs_config_pdu->start_rb + csirs_config_pdu->nr_of_rbs); rb++) {
      if (csirs_config_pdu->freq_density <= 1 &&
          csirs_config_pdu->freq_density != (rb % 2)) {
        continue;
      }
      rb_count++;
      uint16_t k = (frame_parms->first_carrier_offset + rb * NR_NB_SC_PER_RB)
                   % frame_parms->ofdm_symbol_size;
      uint16_t k_offset = k + mem_offset;
      for (int ant_rx = 0; ant_rx < frame_parms->nb_antennas_rx; ant_rx++) {
        const c16_t ch = csi_rs_estimated_channel_freq[ant_rx][0][k_offset];
        signal_power += (int64_t)ch.r * ch.r + (int64_t)ch.i * ch.i;
      }
    }

    // noise_power: calc_power_csirs()로 계산된 per-RB 분산값 (Var(|H_LS - H_interp|))
    //              이미 (nb_antennas_rx * ports)로 나뉜 평균
    // signal_power: Σ_rb |H(k)|² 전체 합 → per-RB 평균으로 정규화 필요
    if (frame_parms->nb_antennas_rx > 0) {
      signal_power /= frame_parms->nb_antennas_rx;
    }
    if (rb_count > 0) {
      signal_power /= rb_count;  // per-RB 평균 (noise_power 스케일과 일치시킴)
    }
    /*
     * csi_rs_estimated_channel_freq는 고정소수점 스케일(log2_re)로 누적/보정된 값이다.
     * N_ports>1 경로는 내부 계산에서 log2_re 보정을 반영하지만,
     * 단일 포트 경로에서 이 보정이 빠지면 SINR가 과대평가된다.
     */
    if (log2_re > 0) {
      signal_power >>= log2_re;
    }

    // 정수 기반 SINR 계산 유지 (기존 OAI CQI 매핑 동작과 동일한 스케일)
    uint32_t sinr_linear = (signal_power > 0) ? (uint32_t)(signal_power / effective_noise_power) : 0;
    *precoded_sinr_dB = dB_fixed(sinr_linear);

    LOG_I(NR_PHY, "PMI: Single port (N_ports=1), SINR computed: sig_power=%lld, noise=%u, rb_count=%d, log2_re=%d, sinr_lin=%u, SINR=%d dB\n",
          (long long)signal_power, effective_noise_power, rb_count, log2_re, sinr_linear, *precoded_sinr_dB);
    return 0;
  }
  
  // 노이즈 파워가 0인 경우 기본값 설정 (디버그용)
  uint32_t effective_noise_power = interference_plus_noise_power;
  if (effective_noise_power == 0) {
    effective_noise_power = 1000; // 기본 노이즈 파워 값
    LOG_I(NR_PHY, "PMI: Using default noise power %u (original was 0)\n", effective_noise_power);
  }

  if(rank_indicator == 0 || rank_indicator == 1) {
    c64_t sum[4] = {0};
    c64_t sum2[4] = {0};
    int64_t tested_precoded_sinr[4] = {0};

    for (int rb = csirs_config_pdu->start_rb; rb < (csirs_config_pdu->start_rb+csirs_config_pdu->nr_of_rbs); rb++) {

      if (csirs_config_pdu->freq_density <= 1 && csirs_config_pdu->freq_density != (rb % 2)) {
        continue;
      }
      uint16_t k = (frame_parms->first_carrier_offset + rb * NR_NB_SC_PER_RB) % frame_parms->ofdm_symbol_size;
      uint16_t k_offset = k + mem_offset;
      for (int ant_rx = 0; ant_rx < frame_parms->nb_antennas_rx; ant_rx++) {
        const c16_t p0 = csi_rs_estimated_channel_freq[ant_rx][0][k_offset];
        const c16_t p1 = csi_rs_estimated_channel_freq[ant_rx][1][k_offset];

        // H_p0 + 1*H_p1 = (H_p0_re + H_p1_re) + 1j*(H_p0_im + H_p1_im)
        sum[0].r += (p0.r + p1.r);
        sum[0].i += (p0.i + p1.i);
        sum2[0].r += (sum[0].r * sum[0].r) >> log2_re;
        sum2[0].i += (sum[0].i * sum[0].i) >> log2_re;

        // H_p0 + 1j*H_p1 = (H_p0_re - H_p1_im) + 1j*(H_p0_im + H_p1_re)
        sum[1].r += (p0.r - p1.i);
        sum[1].i += (p0.i + p1.r);
        sum2[1].r += (sum[1].r * sum[1].r) >> log2_re;
        sum2[1].i += (sum[1].i * sum[1].i) >> log2_re;

        // H_p0 - 1*H_p1 = (H_p0_re - H_p1_re) + 1j*(H_p0_im - H_p1_im)
        sum[2].r += (p0.r - p1.r);
        sum[2].i += (p0.i - p1.i);
        sum2[2].r += (sum[2].r * sum[2].r) >> log2_re;
        sum2[2].i += (sum[2].i * sum[2].i) >> log2_re;

        // H_p0 - 1j*H_p1 = (H_p0_re + H_p1_im) + 1j*(H_p0_im - H_p1_re)
        sum[3].r += (p0.r + p1.i);
        sum[3].i += (p0.i - p1.r);
        sum2[3].r += (sum[3].r * sum[3].r) >> log2_re;
        sum2[3].i += (sum[3].i * sum[3].i) >> log2_re;
      }
    }

    // We should perform >>nr_csi_info->log2_re here for all terms, but since sum2_re and sum2_im can be high values,
    // we performed this above.
    for(int p = 0; p<4; p++) {
      int64_t power_re = sum2[p].r - (sum[p].r >> log2_re) * (sum[p].r >> log2_re);
      int64_t power_im = sum2[p].i - (sum[p].i >> log2_re) * (sum[p].i >> log2_re);
      tested_precoded_sinr[p] = (power_re + power_im) / effective_noise_power;
    }

    if(rank_indicator == 0) {
      for(int tested_i2 = 0; tested_i2 < 4; tested_i2++) {
        if(tested_precoded_sinr[tested_i2] > tested_precoded_sinr[i2[0]]) {
          i2[0] = tested_i2;
        }
      }
      *precoded_sinr_dB = dB_fixed(tested_precoded_sinr[i2[0]]);
    } else {
      i2[0] = tested_precoded_sinr[0]+tested_precoded_sinr[2] > tested_precoded_sinr[1]+tested_precoded_sinr[3] ? 0 : 1;
      *precoded_sinr_dB = dB_fixed((tested_precoded_sinr[i2[0]] + tested_precoded_sinr[i2[0]+2])>>1);
    }
    
    // 디버그 로그 추가
    LOG_I(NR_PHY, "PMI: i2[0]=%d, SINR=%d dB, tested_SINR=[%ld,%ld,%ld,%ld]\n", 
          i2[0], *precoded_sinr_dB, tested_precoded_sinr[0], tested_precoded_sinr[1], 
          tested_precoded_sinr[2], tested_precoded_sinr[3]);

  } else {
    LOG_W(NR_PHY, "PMI computation is not implemented for rank indicator %i\n", rank_indicator+1);
    return -1;
  }

  LOG_I(NR_PHY, "PMI: Function completed successfully\n");
  return 0;
}

int nr_csi_rs_type2_port_selection_pmi(const PHY_VARS_NR_UE *ue,
                                      const fapi_nr_dl_config_csirs_pdu_rel15_t *csirs_config_pdu,
                                      const uint8_t N_ports,
                                      uint8_t mem_offset,
                                      const c16_t csi_rs_estimated_channel_freq[][N_ports][ue->frame_parms.ofdm_symbol_size + FILTER_MARGIN],
                                      const uint32_t interference_plus_noise_power,
                                      const uint8_t rank_indicator,
                                      const int16_t log2_re,
                                      uint8_t port_sel_d,
                                      uint8_t num_beams_L,
                                      uint8_t phase_alphabet,
                                      bool subband_amplitude_flag,
                                      fapi_nr_csirs_measurements_t *meas,
                                      uint32_t *precoded_sinr_dB)
{
  const NR_DL_FRAME_PARMS *fp = &ue->frame_parms;
  const int num_rx = fp->nb_antennas_rx;
  const int num_groups = N_ports / port_sel_d;

  if (num_groups < 1 || port_sel_d < 1 || num_beams_L < 2 || num_beams_L > 4) {
    LOG_E(NR_PHY, "Type-II PortSel: invalid params d=%d, L=%d, N_ports=%d\n",
          port_sel_d, num_beams_L, N_ports);
    return -1;
  }

  meas->is_type2 = true;
  meas->num_beams = num_beams_L;
  meas->port_sel_d = port_sel_d;

  // Step 1: Port selection - find the best group of d contiguous ports
  int64_t group_power[16] = {0};
  int rb_count = 0;
  for (int rb = csirs_config_pdu->start_rb;
       rb < (csirs_config_pdu->start_rb + csirs_config_pdu->nr_of_rbs); rb++) {
    if (csirs_config_pdu->freq_density <= 1 && csirs_config_pdu->freq_density != (rb % 2))
      continue;
    rb_count++;
    uint16_t k = (fp->first_carrier_offset + rb * NR_NB_SC_PER_RB) % fp->ofdm_symbol_size;
    uint16_t k_off = k + mem_offset;
    for (int g = 0; g < num_groups; g++) {
      for (int p = 0; p < port_sel_d; p++) {
        int port_idx = g * port_sel_d + p;
        if (port_idx >= N_ports) break;
        for (int ar = 0; ar < num_rx; ar++) {
          const c16_t h = csi_rs_estimated_channel_freq[ar][port_idx][k_off];
          group_power[g] += (int64_t)h.r * h.r + (int64_t)h.i * h.i;
        }
      }
    }
  }

  int best_group = 0;
  for (int g = 1; g < num_groups; g++) {
    if (group_power[g] > group_power[best_group])
      best_group = g;
  }
  meas->port_selection_indicator = best_group;
  int sel_port_start = best_group * port_sel_d;

  LOG_D(NR_PHY, "Type-II PortSel: best_group=%d (ports %d..%d), power=%lld\n",
        best_group, sel_port_start, sel_port_start + port_sel_d - 1,
        (long long)group_power[best_group]);

  // Step 2: For each layer, compute wideband per-port power for amplitude quantization.
  // Need power for both polarizations: ports [sel_start .. sel_start+d-1] (pol 0)
  // and ports [sel_start+N_ports/2 .. sel_start+N_ports/2+d-1] (pol 1).
  int num_layers = rank_indicator + 1;
  if (num_layers > FAPI_NR_TYPE2_MAX_LAYERS) num_layers = FAPI_NR_TYPE2_MAX_LAYERS;

  int half_ports = N_ports / 2;
  int total_coeffs_max = 2 * num_beams_L;
  if (total_coeffs_max > 2 * port_sel_d) total_coeffs_max = 2 * port_sel_d;

  // Per-coefficient wideband power (includes both polarizations)
  int64_t coeff_power[2 * FAPI_NR_TYPE2_MAX_BEAMS] = {0};
  for (int c = 0; c < total_coeffs_max && c < 2 * FAPI_NR_TYPE2_MAX_BEAMS; c++) {
    int port_local = c % port_sel_d;
    int pol = c / port_sel_d;
    int port_idx = sel_port_start + port_local + pol * half_ports;
    if (port_idx >= N_ports) continue;
    for (int rb = csirs_config_pdu->start_rb;
         rb < (csirs_config_pdu->start_rb + csirs_config_pdu->nr_of_rbs); rb++) {
      if (csirs_config_pdu->freq_density <= 1 && csirs_config_pdu->freq_density != (rb % 2))
        continue;
      uint16_t k = (fp->first_carrier_offset + rb * NR_NB_SC_PER_RB) % fp->ofdm_symbol_size;
      uint16_t k_off = k + mem_offset;
      for (int ar = 0; ar < num_rx; ar++) {
        const c16_t h = csi_rs_estimated_channel_freq[ar][port_idx][k_off];
        coeff_power[c] += ((int64_t)h.r * h.r + (int64_t)h.i * h.i) >> log2_re;
      }
    }
  }

  // Step 3: Quantize wideband amplitudes using per-coefficient power
  for (int lay = 0; lay < num_layers; lay++) {
    int64_t max_power = 0;
    int strongest_idx = 0;
    int total_coeffs = total_coeffs_max;

    for (int c = 0; c < total_coeffs && c < 2 * FAPI_NR_TYPE2_MAX_BEAMS; c++) {
      if (coeff_power[c] > max_power) {
        max_power = coeff_power[c];
        strongest_idx = c;
      }
    }
    meas->strongest_coeff_indicator[lay] = strongest_idx;

    for (int c = 0; c < total_coeffs && c < 2 * FAPI_NR_TYPE2_MAX_BEAMS; c++) {
      if (c == strongest_idx) {
        meas->wideband_amplitude[c][lay] = 0;
        continue;
      }
      if (max_power > 0) {
        int64_t ratio_x1024 = (coeff_power[c] * 1024) / max_power;
        if (ratio_x1024 >= 724) meas->wideband_amplitude[c][lay] = 1;
        else if (ratio_x1024 >= 362) meas->wideband_amplitude[c][lay] = 2;
        else if (ratio_x1024 >= 181) meas->wideband_amplitude[c][lay] = 3;
        else if (ratio_x1024 >= 91) meas->wideband_amplitude[c][lay] = 4;
        else if (ratio_x1024 >= 45) meas->wideband_amplitude[c][lay] = 5;
        else if (ratio_x1024 >= 16) meas->wideband_amplitude[c][lay] = 6;
        else meas->wideband_amplitude[c][lay] = 7;
      } else {
        meas->wideband_amplitude[c][lay] = 7;
      }
    }
  }

  // Step 4: Subband phase/amplitude coefficients
  // Use correct port index with polarization offset: pol0 ports = sel_start + local,
  // pol1 ports = sel_start + local + N_ports/2
  int total_rbs = csirs_config_pdu->nr_of_rbs;
  int subband_size = (total_rbs >= 24) ? ((total_rbs >= 72) ? 8 : 4) : total_rbs;
  int num_sb = (total_rbs + subband_size - 1) / subband_size;
  if (num_sb > FAPI_NR_TYPE2_MAX_SUBBANDS) num_sb = FAPI_NR_TYPE2_MAX_SUBBANDS;
  meas->num_subbands = num_sb;

  {
    int total_coeffs = total_coeffs_max;

    for (int sb = 0; sb < num_sb; sb++) {
      int sb_start_rb = csirs_config_pdu->start_rb + sb * subband_size;
      int sb_end_rb = sb_start_rb + subband_size;
      if (sb_end_rb > csirs_config_pdu->start_rb + total_rbs)
        sb_end_rb = csirs_config_pdu->start_rb + total_rbs;

      for (int lay = 0; lay < num_layers; lay++) {
        for (int c = 0; c < total_coeffs && c < 2 * FAPI_NR_TYPE2_MAX_BEAMS; c++) {
          int port_local = c % port_sel_d;
          int pol = c / port_sel_d;
          int port_idx = sel_port_start + port_local + pol * half_ports;
          if (port_idx >= N_ports) {
            meas->subband_phase[sb][c][lay] = 0;
            meas->subband_amp[sb][c][lay] = 0;
            continue;
          }
          int64_t sb_re = 0, sb_im = 0;

          for (int rb = sb_start_rb; rb < sb_end_rb; rb++) {
            if (csirs_config_pdu->freq_density <= 1 && csirs_config_pdu->freq_density != (rb % 2))
              continue;
            uint16_t k = (fp->first_carrier_offset + rb * NR_NB_SC_PER_RB) % fp->ofdm_symbol_size;
            uint16_t k_off = k + mem_offset;
            for (int ar = 0; ar < num_rx; ar++) {
              const c16_t h = csi_rs_estimated_channel_freq[ar][port_idx][k_off];
              sb_re += h.r;
              sb_im += h.i;
            }
          }

          int phase_idx = 0;
          if (phase_alphabet == 4) {
            if (sb_re >= 0 && sb_im >= 0) phase_idx = 0;
            else if (sb_re < 0 && sb_im >= 0) phase_idx = 1;
            else if (sb_re < 0 && sb_im < 0) phase_idx = 2;
            else phase_idx = 3;
          } else {
            int64_t abs_re = sb_re < 0 ? -sb_re : sb_re;
            int64_t abs_im = sb_im < 0 ? -sb_im : sb_im;
            int octant;
            if (abs_im <= (abs_re * 4142 / 10000))
              octant = 0;
            else if (abs_re <= (abs_im * 4142 / 10000))
              octant = 2;
            else
              octant = 1;
            if (sb_re >= 0 && sb_im >= 0) phase_idx = octant;
            else if (sb_re < 0 && sb_im >= 0) phase_idx = 4 - octant;
            else if (sb_re < 0 && sb_im < 0) phase_idx = 4 + octant;
            else phase_idx = 8 - octant;
            phase_idx &= 0x7;
          }
          meas->subband_phase[sb][c][lay] = phase_idx;

          if (subband_amplitude_flag) {
            int64_t sb_power = sb_re * sb_re + sb_im * sb_im;
            int sb_rb_count = sb_end_rb - sb_start_rb;
            int64_t threshold = (coeff_power[c] * sb_rb_count) / (2 * (rb_count > 0 ? rb_count : 1));
            meas->subband_amp[sb][c][lay] = (sb_power >= threshold) ? 0 : 1;
          } else {
            meas->subband_amp[sb][c][lay] = 0;
          }
        }
      }
    }
  }

  // Step 5: Compute precoded SINR for CQI
  uint32_t eff_noise = interference_plus_noise_power > 0 ? interference_plus_noise_power : 1;
  int64_t signal_power = group_power[best_group];
  if (rb_count > 0) signal_power /= rb_count;
  if (num_rx > 0) signal_power /= num_rx;
  if (log2_re > 0) signal_power >>= log2_re;

  uint32_t sinr_linear = (signal_power > 0) ? (uint32_t)(signal_power / eff_noise) : 0;
  *precoded_sinr_dB = dB_fixed(sinr_linear);

  LOG_I(NR_PHY, "Type-II PortSel PMI: group=%d, L=%d, d=%d, layers=%d, SINR=%d dB, subbands=%d\n",
        best_group, num_beams_L, port_sel_d, num_layers, *precoded_sinr_dB, num_sb);
  return 0;
}

int nr_csi_rs_cqi_estimation(const uint32_t precoded_sinr,
                             uint8_t *cqi) {

  *cqi = 0;

  // Default SINR table for an AWGN channel for SISO scenario, considering 0.1 BLER condition and TS 38.214 Table 5.2.2.1-2
  if(precoded_sinr>0 && precoded_sinr<=2) {
    *cqi = 4;
  } else if(precoded_sinr==3) {
    *cqi = 5;
  } else if(precoded_sinr>3 && precoded_sinr<=5) {
    *cqi = 6;
  } else if(precoded_sinr>5 && precoded_sinr<=7) {
    *cqi = 7;
  } else if(precoded_sinr>7 && precoded_sinr<=9) {
    *cqi = 8;
  } else if(precoded_sinr==10) {
    *cqi = 9;
  } else if(precoded_sinr>10 && precoded_sinr<=12) {
    *cqi = 10;
  } else if(precoded_sinr>12 && precoded_sinr<=15) {
    *cqi = 11;
  } else if(precoded_sinr==16) {
    *cqi = 12;
  } else if(precoded_sinr>16 && precoded_sinr<=18) {
    *cqi = 13;
  } else if(precoded_sinr==19) {
    *cqi = 14;
  } else if(precoded_sinr>19) {
    *cqi = 15;
  }

  // 디버그 로그 추가
  LOG_I(NR_PHY, "CQI: SINR=%d dB -> CQI=%d\n", precoded_sinr, *cqi);

  return 0;
}

static void nr_csi_im_power_estimation(const PHY_VARS_NR_UE *ue,
                                       const UE_nr_rxtx_proc_t *proc,
                                       const fapi_nr_dl_config_csiim_pdu_rel15_t *csiim_config_pdu,
                                       uint32_t *interference_plus_noise_power,
                                       const c16_t rxdataF[][ue->frame_parms.samples_per_slot_wCP])
{
  const NR_DL_FRAME_PARMS *frame_parms = &ue->frame_parms;

  const uint16_t end_rb = csiim_config_pdu->start_rb + csiim_config_pdu->nr_of_rbs > csiim_config_pdu->bwp_size ?
                          csiim_config_pdu->bwp_size : csiim_config_pdu->start_rb + csiim_config_pdu->nr_of_rbs;

  int32_t count = 0;
  int32_t sum_re = 0;
  int32_t sum_im = 0;
  int32_t sum2_re = 0;
  int32_t sum2_im = 0;

  int l_csiim[4] = {-1, -1, -1, -1};

  for(int symb_idx = 0; symb_idx < 4; symb_idx++) {

    uint8_t symb = csiim_config_pdu->l_csiim[symb_idx];
    bool done = false;
    for (int symb_idx2 = 0; symb_idx2 < symb_idx; symb_idx2++) {
      if (l_csiim[symb_idx2] == symb) {
        done = true;
      }
    }

    if (done) {
      continue;
    }

    l_csiim[symb_idx] = symb;
    uint64_t symbol_offset = symb*frame_parms->ofdm_symbol_size;

    for (int ant_rx = 0; ant_rx < frame_parms->nb_antennas_rx; ant_rx++) {

      const c16_t *rx_signal = &rxdataF[ant_rx][symbol_offset];

      for (int rb = csiim_config_pdu->start_rb; rb < end_rb; rb++) {

        uint16_t sc0_offset = (frame_parms->first_carrier_offset + rb*NR_NB_SC_PER_RB) % frame_parms->ofdm_symbol_size;

        for (int sc_idx = 0; sc_idx < 4; sc_idx++) {

          uint16_t sc = sc0_offset + csiim_config_pdu->k_csiim[sc_idx];
          if (sc >= frame_parms->ofdm_symbol_size) {
            sc -= frame_parms->ofdm_symbol_size;
          }

#ifdef NR_CSIIM_DEBUG
          LOG_I(NR_PHY, "(ant_rx %i, sc %i) real %i, imag %i\n", ant_rx, sc, rx_signal[sc].r, rx_signal[sc].i);
#endif

          sum_re += rx_signal[sc].r;
          sum_im += rx_signal[sc].i;
          sum2_re += rx_signal[sc].r * rx_signal[sc].r;
          sum2_im += rx_signal[sc].i * rx_signal[sc].i;
          count++;
        }
      }
    }
  }

  int32_t power_re = sum2_re / count - (sum_re / count) * (sum_re / count);
  int32_t power_im = sum2_im / count - (sum_im / count) * (sum_im / count);

  *interference_plus_noise_power = power_re + power_im;

#ifdef NR_CSIIM_DEBUG
  LOG_I(NR_PHY, "interference_plus_noise_power based on CSI-IM = %i\n", *interference_plus_noise_power);
#endif
}

void nr_ue_csi_im_procedures(PHY_VARS_NR_UE *ue,
                             const UE_nr_rxtx_proc_t *proc,
                             const c16_t rxdataF[][ue->frame_parms.samples_per_slot_wCP],
                             const fapi_nr_dl_config_csiim_pdu_rel15_t *csiim_config_pdu)
{

#ifdef NR_CSIIM_DEBUG
  LOG_I(NR_PHY, "csiim_config_pdu->bwp_size = %i\n", csiim_config_pdu->bwp_size);
  LOG_I(NR_PHY, "csiim_config_pdu->bwp_start = %i\n", csiim_config_pdu->bwp_start);
  LOG_I(NR_PHY, "csiim_config_pdu->subcarrier_spacing = %i\n", csiim_config_pdu->subcarrier_spacing);
  LOG_I(NR_PHY, "csiim_config_pdu->start_rb = %i\n", csiim_config_pdu->start_rb);
  LOG_I(NR_PHY, "csiim_config_pdu->nr_of_rbs = %i\n", csiim_config_pdu->nr_of_rbs);
  LOG_I(NR_PHY, "csiim_config_pdu->k_csiim = %i.%i.%i.%i\n", csiim_config_pdu->k_csiim[0], csiim_config_pdu->k_csiim[1], csiim_config_pdu->k_csiim[2], csiim_config_pdu->k_csiim[3]);
  LOG_I(NR_PHY, "csiim_config_pdu->l_csiim = %i.%i.%i.%i\n", csiim_config_pdu->l_csiim[0], csiim_config_pdu->l_csiim[1], csiim_config_pdu->l_csiim[2], csiim_config_pdu->l_csiim[3]);
#endif

  nr_csi_im_power_estimation(ue, proc, csiim_config_pdu, &ue->nr_csi_info->interference_plus_noise_power, rxdataF);
  ue->nr_csi_info->csi_im_meas_computed = true;
}

void nr_ue_csi_rs_procedures(PHY_VARS_NR_UE *ue,
                             const UE_nr_rxtx_proc_t *proc,
                             const c16_t rxdataF[][ue->frame_parms.samples_per_slot_wCP],
                             fapi_nr_dl_config_csirs_pdu_rel15_t *csirs_config_pdu)
{

#ifdef NR_CSIRS_DEBUG
  LOG_I(NR_PHY, "csirs_config_pdu->subcarrier_spacing = %i\n", csirs_config_pdu->subcarrier_spacing);
  LOG_I(NR_PHY, "csirs_config_pdu->cyclic_prefix = %i\n", csirs_config_pdu->cyclic_prefix);
  LOG_I(NR_PHY, "csirs_config_pdu->start_rb = %i\n", csirs_config_pdu->start_rb);
  LOG_I(NR_PHY, "csirs_config_pdu->nr_of_rbs = %i\n", csirs_config_pdu->nr_of_rbs);
  LOG_I(NR_PHY, "csirs_config_pdu->csi_type = %i (0:TRS, 1:CSI-RS NZP, 2:CSI-RS ZP)\n", csirs_config_pdu->csi_type);
  LOG_I(NR_PHY, "csirs_config_pdu->row = %i\n", csirs_config_pdu->row);
  LOG_I(NR_PHY, "csirs_config_pdu->freq_domain = %i\n", csirs_config_pdu->freq_domain);
  LOG_I(NR_PHY, "csirs_config_pdu->symb_l0 = %i\n", csirs_config_pdu->symb_l0);
  LOG_I(NR_PHY, "csirs_config_pdu->symb_l1 = %i\n", csirs_config_pdu->symb_l1);
  LOG_I(NR_PHY, "csirs_config_pdu->cdm_type = %i\n", csirs_config_pdu->cdm_type);
  LOG_I(NR_PHY, "csirs_config_pdu->freq_density = %i (0: dot5 (even RB), 1: dot5 (odd RB), 2: one, 3: three)\n", csirs_config_pdu->freq_density);
  LOG_I(NR_PHY, "csirs_config_pdu->scramb_id = %i\n", csirs_config_pdu->scramb_id);
  LOG_I(NR_PHY, "csirs_config_pdu->power_control_offset = %i\n", csirs_config_pdu->power_control_offset);
  LOG_I(NR_PHY, "csirs_config_pdu->power_control_offset_ss = %i\n", csirs_config_pdu->power_control_offset_ss);
#endif

  if(csirs_config_pdu->csi_type == 0) {
    LOG_E(NR_PHY, "Handling of CSI-RS for tracking not handled yet at PHY\n");
    return;
  }

  if(csirs_config_pdu->csi_type == 2) {
    LOG_E(NR_PHY, "Handling of ZP CSI-RS not handled yet at PHY\n");
    return;
  }

  const NR_DL_FRAME_PARMS *frame_parms = &ue->frame_parms;
  csi_mapping_parms_t mapping_parms = get_csi_mapping_parms(csirs_config_pdu->row,
                                                            csirs_config_pdu->freq_domain,
                                                            csirs_config_pdu->symb_l0,
                                                            csirs_config_pdu->symb_l1);
  nr_csi_info_t *csi_info = ue->nr_csi_info;
  nr_generate_csi_rs(frame_parms,
                     &mapping_parms,
                     AMP,
                     proc->nr_slot_rx,
                     csirs_config_pdu->freq_density,
                     csirs_config_pdu->start_rb,
                     csirs_config_pdu->nr_of_rbs,
                     csirs_config_pdu->symb_l0,
                     csirs_config_pdu->symb_l1,
                     csirs_config_pdu->row,
                     csirs_config_pdu->scramb_id,
                     csirs_config_pdu->power_control_offset_ss,
                     csirs_config_pdu->cdm_type,
                     csi_info->csi_rs_generated_signal);

  csi_info->csi_rs_generated_signal_bits = log2_approx(AMP);

  c16_t csi_rs_ls_estimated_channel[frame_parms->nb_antennas_rx][mapping_parms.ports][frame_parms->ofdm_symbol_size];
  c16_t csi_rs_estimated_channel_freq[frame_parms->nb_antennas_rx][mapping_parms.ports]
                                     [frame_parms->ofdm_symbol_size + FILTER_MARGIN];

  // (long)&csi_rs_estimated_channel_freq[0][0][frame_parms->first_carrier_offset] & 0x1F
  // gives us the remainder of the integer division by 32 of the memory address
  // By subtracting the previous value of 32, we know how much is left to have a multiple of 32.
  // Doing >> 2 <=> /sizeof(int32_t), we know what is the index offset of the array.
  uint8_t mem_offset = (((32 - ((long)&csi_rs_estimated_channel_freq[0][0][frame_parms->first_carrier_offset])) & 0x1F) >> 2);
  int CDM_group_size = get_cdm_group_size(csirs_config_pdu->cdm_type);
  c16_t csi_rs_received_signal[frame_parms->nb_antennas_rx][frame_parms->samples_per_slot_wCP];
  uint32_t rsrp = 0;
  int rsrp_dBm = 0;
  nr_get_csi_rs_signal(ue,
                       proc,
                       csirs_config_pdu,
                       csi_info,
                       &mapping_parms,
                       CDM_group_size,
                       csi_rs_received_signal,
                       &rsrp,
                       &rsrp_dBm,
                       rxdataF);


  uint32_t noise_power = 0;
  int16_t log2_re = 0;
  int16_t log2_maxh = 0;
  // if we need to measure only RSRP no need to do channel estimation
  if (csirs_config_pdu->measurement_bitmap > 1)
    nr_csi_rs_channel_estimation(frame_parms,
                                 proc,
                                 csirs_config_pdu,
                                 csi_info,
                                 (const c16_t **)csi_info->csi_rs_generated_signal,
                                 csi_rs_received_signal,
                                 &mapping_parms,
                                 CDM_group_size,
                                 mem_offset,
                                 csi_rs_ls_estimated_channel,
                                 csi_rs_estimated_channel_freq,
                                 &log2_re,
                                 &log2_maxh,
                                 &noise_power);

  uint8_t rank_indicator = 0;
  // bit 1 in bitmap to indicate RI measurment
  if (csirs_config_pdu->measurement_bitmap & 2) {
    nr_csi_rs_ri_estimation(ue,
                            csirs_config_pdu,
                            csi_info,
                            mapping_parms.ports,
                            mem_offset,
                            csi_rs_estimated_channel_freq,
                            log2_maxh,
                            &rank_indicator);
  }

  uint8_t i1[3] = {0};
  uint8_t i2[1] = {0};
  uint8_t cqi = 0;
  uint32_t precoded_sinr_dB = 0;
  fapi_nr_csirs_measurements_t csirs_measurements;
  memset(&csirs_measurements, 0, sizeof(csirs_measurements));
  csirs_measurements.is_type2 = false;

  // bit 3 in bitmap to indicate PMI measurement
  if (csirs_config_pdu->measurement_bitmap & 8) {
    uint32_t noise_used = csi_info->csi_im_meas_computed ? csi_info->interference_plus_noise_power : noise_power;
    LOG_I(NR_PHY, "PMI: Calling PMI estimation with ports=%d, noise_power=%u, type2=%d\n",
          mapping_parms.ports, noise_used, csirs_config_pdu->type2_port_selection);

    if (csirs_config_pdu->type2_port_selection && mapping_parms.ports > 2) {
      nr_csi_rs_type2_port_selection_pmi(ue,
                                         csirs_config_pdu,
                                         mapping_parms.ports,
                                         mem_offset,
                                         csi_rs_estimated_channel_freq,
                                         noise_used,
                                         rank_indicator,
                                         log2_re,
                                         csirs_config_pdu->type2_port_sel_d,
                                         csirs_config_pdu->type2_num_beams,
                                         csirs_config_pdu->type2_phase_alphabet,
                                         csirs_config_pdu->type2_subband_amplitude,
                                         &csirs_measurements,
                                         &precoded_sinr_dB);
    } else {
      nr_csi_rs_pmi_estimation(ue,
                               csirs_config_pdu,
                               csi_info,
                               mapping_parms.ports,
                               mem_offset,
                               csi_rs_estimated_channel_freq,
                               noise_used,
                               rank_indicator,
                               log2_re,
                               i1,
                               i2,
                               &precoded_sinr_dB);
    }

    if(csirs_config_pdu->measurement_bitmap & 16)
      nr_csi_rs_cqi_estimation(precoded_sinr_dB, &cqi);
  }

  LOG_I(NR_PHY, "[UE %d] CSI-RS Results: RSRP=%d dBm, RI=%d, type2=%d, SINR=%d dB, CQI=%d\n",
        ue->Mod_id, rsrp_dBm, rank_indicator + 1, csirs_measurements.is_type2,
        precoded_sinr_dB, cqi);

  csirs_measurements.rsrp = rsrp;
  csirs_measurements.rsrp_dBm = rsrp_dBm;
  csirs_measurements.rank_indicator = rank_indicator;
  csirs_measurements.i1 = *i1;
  csirs_measurements.i2 = *i2;
  csirs_measurements.cqi = cqi;
  csirs_measurements.radiolink_monitoring = RLM_no_monitoring;

  {
    static int csi_log_init = 0;
    static FILE *csi_log_fp = NULL;
    if (!csi_log_init) {
      csi_log_init = 1;
      const char *lp = getenv("CSI_CHANNEL_LOG");
      if (lp && lp[0]) {
        csi_log_fp = fopen(lp, "wb");
        if (csi_log_fp)
          LOG_I(NR_PHY, "CSI Channel Log: writing to %s\n", lp);
        else
          LOG_E(NR_PHY, "CSI Channel Log: cannot open %s\n", lp);
      }
    }
    if (csi_log_fp) {
      const uint16_t np = mapping_parms.ports;
      const uint16_t nr = frame_parms->nb_antennas_rx;
      const uint32_t os = frame_parms->ofdm_symbol_size;
      const uint8_t pt = csirs_measurements.is_type2 ? 1 : 0;

      int tc = pt ? 2 * csirs_measurements.num_beams : 0;
      if (tc > 2 * FAPI_NR_TYPE2_MAX_BEAMS) tc = 2 * FAPI_NR_TYPE2_MAX_BEAMS;
      int nl = rank_indicator + 1;
      if (nl > FAPI_NR_TYPE2_MAX_LAYERS) nl = FAPI_NR_TYPE2_MAX_LAYERS;

      uint32_t ch_bytes = nr * np * os * sizeof(c16_t);
      uint32_t sb_bytes = pt ? (csirs_measurements.num_subbands * tc * nl) : 0;
      uint32_t rec_size = 64 + ch_bytes + sb_bytes;

      uint8_t hdr[64];
      memset(hdr, 0, 64);
      uint32_t magic = 0x43534932;
      memcpy(hdr +  0, &magic, 4);
      memcpy(hdr +  4, &rec_size, 4);
      uint32_t fv = (uint32_t)proc->frame_rx;
      uint32_t sv = (uint32_t)proc->nr_slot_rx;
      memcpy(hdr +  8, &fv, 4);
      memcpy(hdr + 12, &sv, 4);
      memcpy(hdr + 16, &np, 2);
      memcpy(hdr + 18, &nr, 2);
      memcpy(hdr + 20, &os, 4);
      uint32_t fc = frame_parms->first_carrier_offset;
      memcpy(hdr + 24, &fc, 4);
      uint16_t srb = csirs_config_pdu->start_rb;
      uint16_t nrb = csirs_config_pdu->nr_of_rbs;
      memcpy(hdr + 28, &srb, 2);
      memcpy(hdr + 30, &nrb, 2);
      hdr[32] = mem_offset;
      hdr[33] = rank_indicator;
      hdr[34] = cqi;
      hdr[35] = pt;
      hdr[36] = i1[0];
      hdr[37] = i2[0];
      hdr[38] = csirs_measurements.port_selection_indicator;
      hdr[39] = csirs_measurements.strongest_coeff_indicator[0];
      hdr[40] = (nl > 1) ? csirs_measurements.strongest_coeff_indicator[1] : 0;
      hdr[41] = csirs_measurements.num_beams;
      hdr[42] = csirs_measurements.port_sel_d;
      hdr[43] = csirs_config_pdu->type2_phase_alphabet;
      for (int c = 0; c < 8 && c < tc; c++)
        hdr[44 + c] = csirs_measurements.wideband_amplitude[c][0];
      for (int c = 0; c < 8 && c < tc; c++)
        hdr[52 + c] = (nl > 1) ? csirs_measurements.wideband_amplitude[c][1] : 0;
      hdr[60] = csirs_measurements.num_subbands;

      fwrite(hdr, 1, 64, csi_log_fp);
      for (int arx = 0; arx < nr; arx++)
        for (int pp = 0; pp < np; pp++)
          fwrite(&csi_rs_estimated_channel_freq[arx][pp][0], sizeof(c16_t), os, csi_log_fp);

      if (sb_bytes > 0) {
        for (int sb = 0; sb < csirs_measurements.num_subbands; sb++)
          for (int cc = 0; cc < tc; cc++)
            for (int ll = 0; ll < nl; ll++)
              fwrite(&csirs_measurements.subband_phase[sb][cc][ll], 1, 1, csi_log_fp);
      }
      fflush(csi_log_fp);
    }
  }

  nr_downlink_indication_t dl_indication;
  fapi_nr_rx_indication_t rx_ind = {0};
  nr_fill_dl_indication(&dl_indication, NULL, &rx_ind, proc, ue, NULL);
  nr_fill_rx_indication(&rx_ind, FAPI_NR_CSIRS_IND, ue, NULL, NULL, 1, proc, (void *)&csirs_measurements, NULL);
  if (ue->if_inst && ue->if_inst->dl_indication)
    ue->if_inst->dl_indication(&dl_indication);
}
