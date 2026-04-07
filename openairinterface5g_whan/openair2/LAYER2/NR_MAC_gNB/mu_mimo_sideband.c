#include "mu_mimo_sideband.h"
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

static FILE *sched_fp = NULL;
static FILE *harq_fp  = NULL;
static pthread_mutex_t sched_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t harq_lock  = PTHREAD_MUTEX_INITIALIZER;
static int sched_lines = 0;
static int harq_lines  = 0;
#define FLUSH_INTERVAL 50

void mu_mimo_sideband_init(const char *log_dir)
{
  char path[512];

  if (log_dir && log_dir[0]) {
    mkdir(log_dir, 0755);
  }

  snprintf(path, sizeof(path), "%s/mu_mimo_sched.csv",
           (log_dir && log_dir[0]) ? log_dir : ".");
  sched_fp = fopen(path, "w");
  if (sched_fp) {
    fprintf(sched_fp,
            "frame,slot,rnti,pm_index,pmi_x1,pmi_x2,is_type2,"
            "mcs,rb_start,rb_size,n_layers,cqi,ri,"
            "is_mu_mimo,paired_rnti,is_secondary,is_retx,harq_round,tb_size\n");
    fflush(sched_fp);
  }

  snprintf(path, sizeof(path), "%s/mu_mimo_harq.csv",
           (log_dir && log_dir[0]) ? log_dir : ".");
  harq_fp = fopen(path, "w");
  if (harq_fp) {
    fprintf(harq_fp, "frame,slot,rnti,harq_pid,ack,harq_round\n");
    fflush(harq_fp);
  }
}

void mu_mimo_sideband_log_dl(const mu_mimo_sched_entry_t *e)
{
  if (!sched_fp || !e)
    return;

  pthread_mutex_lock(&sched_lock);
  fprintf(sched_fp,
          "%d,%d,0x%04x,%u,%u,%u,%d,"
          "%u,%u,%u,%u,%u,%u,"
          "%d,0x%04x,%d,%d,%u,%d\n",
          e->frame, e->slot, e->rnti, e->pm_index, e->pmi_x1, e->pmi_x2, e->is_type2,
          e->mcs, e->rb_start, e->rb_size, e->n_layers, e->cqi, e->ri,
          e->is_mu_mimo, e->paired_rnti, e->is_secondary, e->is_retx, e->harq_round, e->tb_size);

  if (++sched_lines % FLUSH_INTERVAL == 0)
    fflush(sched_fp);
  pthread_mutex_unlock(&sched_lock);
}

void mu_mimo_sideband_log_harq(const mu_mimo_harq_entry_t *e)
{
  if (!harq_fp || !e)
    return;

  pthread_mutex_lock(&harq_lock);
  fprintf(harq_fp, "%d,%d,0x%04x,%d,%d,%u\n",
          e->frame, e->slot, e->rnti, e->harq_pid, e->ack, e->harq_round);

  if (++harq_lines % FLUSH_INTERVAL == 0)
    fflush(harq_fp);
  pthread_mutex_unlock(&harq_lock);
}

void mu_mimo_sideband_close(void)
{
  pthread_mutex_lock(&sched_lock);
  if (sched_fp) { fflush(sched_fp); fclose(sched_fp); sched_fp = NULL; }
  pthread_mutex_unlock(&sched_lock);

  pthread_mutex_lock(&harq_lock);
  if (harq_fp) { fflush(harq_fp); fclose(harq_fp); harq_fp = NULL; }
  pthread_mutex_unlock(&harq_lock);
}
