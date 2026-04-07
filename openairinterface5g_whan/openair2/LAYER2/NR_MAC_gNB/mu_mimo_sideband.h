#ifndef MU_MIMO_SIDEBAND_H
#define MU_MIMO_SIDEBAND_H

#include <stdbool.h>
#include <stdint.h>

typedef struct {
  int      frame;
  int      slot;
  uint16_t rnti;
  uint16_t pm_index;
  uint8_t  mcs;
  uint16_t rb_start;
  uint16_t rb_size;
  uint8_t  n_layers;
  uint8_t  cqi;
  uint8_t  ri;
  bool     is_mu_mimo;
  uint16_t paired_rnti;
  bool     is_secondary;
  bool     is_retx;
  uint8_t  harq_round;
  int      tb_size;
  bool     is_type2;
  uint8_t  pmi_x1;
  uint8_t  pmi_x2;
} mu_mimo_sched_entry_t;

typedef struct {
  int      frame;
  int      slot;
  uint16_t rnti;
  bool     ack;
  uint8_t  harq_round;
  int8_t   harq_pid;
} mu_mimo_harq_entry_t;

void mu_mimo_sideband_init(const char *log_dir);
void mu_mimo_sideband_log_dl(const mu_mimo_sched_entry_t *entry);
void mu_mimo_sideband_log_harq(const mu_mimo_harq_entry_t *entry);
void mu_mimo_sideband_close(void);

#endif
