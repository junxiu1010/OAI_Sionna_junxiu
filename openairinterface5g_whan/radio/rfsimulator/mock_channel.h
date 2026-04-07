/*
 * Mock Channel Header for OAI
 * 
 * Header file for the bypass channel module
 * 
 * Author: Minsoo Kim
 * Date: 2024-07-16
 */

#pragma once
#include "openair1/PHY/TOOLS/tools_defs.h"

int mock_channel_init(void);

int mock_channel_start(void);

void mock_channel_stop(void); 